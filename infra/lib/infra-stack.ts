import * as cdk from 'aws-cdk-lib/core';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as autoscaling from 'aws-cdk-lib/aws-autoscaling';
import { Construct } from 'constructs';

export interface GpuClusterProps extends cdk.StackProps {
  /** EC2 instance type for GPU nodes (default: g6e.xlarge) */
  readonly instanceType?: string;
  /** Number of GPU instances in the ASG (default: 1) */
  readonly desiredCapacity?: number;
  /** Maximum number of GPU instances (default: desiredCapacity) */
  readonly maxCapacity?: number;
  /** Number of GPUs to reserve per ECS task (default: 1) */
  readonly gpuCount?: number;
  /** Container image for the inference task */
  readonly containerImage?: string;
  /** Container port for the inference server (default: 8000) */
  readonly containerPort?: number;
  /** Task CPU units (default: 4096) */
  readonly taskCpu?: number;
  /** Task memory MiB (default: 16384) */
  readonly taskMemoryMiB?: number;
}

export class InfraStack extends cdk.Stack {
  public readonly cluster: ecs.Cluster;

  constructor(scope: Construct, id: string, props?: GpuClusterProps) {
    super(scope, id, props);

    const instanceType = props?.instanceType ?? 'g6e.xlarge';
    const desiredCapacity = props?.desiredCapacity ?? 1;
    const maxCapacity = props?.maxCapacity ?? desiredCapacity;
    const gpuCount = props?.gpuCount ?? 1;
    const containerImage = props?.containerImage ?? 'vllm/vllm-openai:latest';
    const containerPort = props?.containerPort ?? 8000;
    const taskCpu = props?.taskCpu ?? 4096;
    const taskMemoryMiB = props?.taskMemoryMiB ?? 16384;

    // VPC with public subnets only (Phase 1 simplicity)
    const vpc = new ec2.Vpc(this, 'Vpc', {
      maxAzs: 2,
      subnetConfiguration: [
        { name: 'Public', subnetType: ec2.SubnetType.PUBLIC, cidrMask: 24 },
      ],
    });

    // ECS Cluster
    this.cluster = new ecs.Cluster(this, 'GpuCluster', { vpc });

    // Auto Scaling Group with ECS-optimized GPU AMI
    const asg = new autoscaling.AutoScalingGroup(this, 'GpuAsg', {
      vpc,
      instanceType: new ec2.InstanceType(instanceType),
      machineImage: ecs.EcsOptimizedImage.amazonLinux2023(
        ecs.AmiHardwareType.GPU,
      ),
      desiredCapacity,
      maxCapacity,
      minCapacity: 0,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      associatePublicIpAddress: true,
    });

    // Allow inbound on inference port
    asg.connections.allowFromAnyIpv4(ec2.Port.tcp(containerPort));

    // Capacity provider with managed scaling
    const capacityProvider = new ecs.AsgCapacityProvider(
      this,
      'GpuCapacityProvider',
      {
        autoScalingGroup: asg,
        enableManagedScaling: true,
        enableManagedTerminationProtection: false,
      },
    );
    this.cluster.addAsgCapacityProvider(capacityProvider);

    // Task definition with GPU resource
    const taskDef = new ecs.Ec2TaskDefinition(this, 'InferenceTask', {
      networkMode: ecs.NetworkMode.HOST,
    });

    taskDef.addContainer('inference', {
      image: ecs.ContainerImage.fromRegistry(containerImage),
      memoryLimitMiB: taskMemoryMiB,
      cpu: taskCpu,
      gpuCount,
      portMappings: [{ containerPort, hostPort: containerPort }],
      logging: ecs.LogDrivers.awsLogs({ streamPrefix: 'gpu-inference' }),
    });

    // Outputs
    new cdk.CfnOutput(this, 'ClusterArn', {
      value: this.cluster.clusterArn,
    });
    new cdk.CfnOutput(this, 'AsgName', {
      value: asg.autoScalingGroupName,
    });
    new cdk.CfnOutput(this, 'TaskDefinitionArn', {
      value: taskDef.taskDefinitionArn,
    });
  }
}

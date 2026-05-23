import * as cdk from 'aws-cdk-lib/core';
import { Template, Match } from 'aws-cdk-lib/assertions';
import { InfraStack } from '../lib/infra-stack';

describe('GpuClusterStack', () => {
  let template: Template;

  beforeAll(() => {
    const app = new cdk.App();
    const stack = new InfraStack(app, 'TestStack');
    template = Template.fromStack(stack);
  });

  test('creates a VPC', () => {
    template.resourceCountIs('AWS::EC2::VPC', 1);
  });

  test('creates an ECS cluster', () => {
    template.resourceCountIs('AWS::ECS::Cluster', 1);
  });

  test('creates an ASG with GPU instance type', () => {
    template.hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
      InstanceType: 'g6e.xlarge',
    });
  });

  test('creates a capacity provider', () => {
    template.resourceCountIs('AWS::ECS::CapacityProvider', 1);
  });

  test('creates a task definition with GPU', () => {
    template.hasResourceProperties('AWS::ECS::TaskDefinition', {
      ContainerDefinitions: Match.arrayWith([
        Match.objectLike({
          Name: 'inference',
          Essential: true,
          Cpu: 4096,
          Memory: 16384,
          ResourceRequirements: [{ Type: 'GPU', Value: '1' }],
        }),
      ]),
    });
  });

  test('respects custom instance type and GPU count', () => {
    const app = new cdk.App();
    const stack = new InfraStack(app, 'CustomStack', {
      instanceType: 'g6e.2xlarge',
      gpuCount: 2,
    });
    const t = Template.fromStack(stack);
    t.hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
      InstanceType: 'g6e.2xlarge',
    });
    t.hasResourceProperties('AWS::ECS::TaskDefinition', {
      ContainerDefinitions: Match.arrayWith([
        Match.objectLike({
          ResourceRequirements: [{ Type: 'GPU', Value: '2' }],
        }),
      ]),
    });
  });
});

#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib/core';
import { InfraStack } from '../lib/infra-stack';

const app = new cdk.App();

new InfraStack(app, 'GpuClusterStack', {
  instanceType: app.node.tryGetContext('instanceType') ?? undefined,
  desiredCapacity: numberOrUndefined(app.node.tryGetContext('desiredCapacity')),
  maxCapacity: numberOrUndefined(app.node.tryGetContext('maxCapacity')),
  gpuCount: numberOrUndefined(app.node.tryGetContext('gpuCount')),
  containerImage: app.node.tryGetContext('containerImage') ?? undefined,
  containerPort: numberOrUndefined(app.node.tryGetContext('containerPort')),
  taskCpu: numberOrUndefined(app.node.tryGetContext('taskCpu')),
  taskMemoryMiB: numberOrUndefined(app.node.tryGetContext('taskMemoryMiB')),
});

function numberOrUndefined(v: unknown): number | undefined {
  return v != null ? Number(v) : undefined;
}

// SPDX-License-Identifier: Apache-2.0
import "slangpy";

float add(float a, float b)
{
    return a + b;
}

[Differentiable]
float3 addt3(float3 a, float3 b)
{
    return a + b;
}

[Differentiable]
float[5] funct5(float[5] a, float[5] b)
{
    float[5] c;
    for (int i = 0; i < 5; i++)
    {
        c[i] = 2.0*a[i] + b[i];
    }
    return c;
}

int2 retId(int2 threadId)
{
    return threadId;
}

[Differentiable]
float2 addt2withId(float2 a,int2 threadId)
{
    return a + (float2)threadId;
}

[Differentiable]
float[48] testNdArray(float [48] a)
{
    return a;
}
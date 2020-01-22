using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

using UnityEngine;

[ExecuteInEditMode]
public class PhysicalConstraint : MonoBehaviour
{
    public bool burst = true;
    public Color wireColor = Color.red;
    [SerializeField] MeshFilter filter;
    [SerializeField] Renderer ren;
    [SerializeField] Mesh mesh;

    [SerializeField] Transform projectTarget;
    [SerializeField] Transform minTarget;
    [SerializeField] Transform maxTarget;
    private void OnEnable()
    {
        filter = GetComponent<MeshFilter>();
        ren = GetComponent<Renderer>();
        mesh = new Mesh();
        CombineInstance[] combineInstance = new CombineInstance[1];
        combineInstance[0].mesh = filter.sharedMesh;
        combineInstance[0].transform = Matrix4x4.identity;
        mesh.CombineMeshes(combineInstance);
    }

    void Update()
    {
        Vector3[] points = filter.sharedMesh.vertices;
        Vector3[] normals = filter.sharedMesh.normals;
        Vector4[] tangents = filter.sharedMesh.tangents;

        if (burst)
        {            
            NativeArray<Vector3> pointsBuffer = new NativeArray<Vector3>(points.Length, Allocator.TempJob);
            NativeArray<Vector3> normalsBuffer = new NativeArray<Vector3>(normals.Length, Allocator.TempJob);
            NativeArray<Vector4> tangentsBuffer = new NativeArray<Vector4>(tangents.Length, Allocator.TempJob);

            NativeArray<Vector3> vectorBuffer = new NativeArray<Vector3>(points.Length, Allocator.TempJob);
            NativeArray<float> relativeDotBuffer = new NativeArray<float>(points.Length, Allocator.TempJob);

            pointsBuffer.CopyFrom(points);
            normalsBuffer.CopyFrom(normals);
            tangentsBuffer.CopyFrom(tangents);

            JobWork jobWork = new JobWork
            {
                pointsBuffer = pointsBuffer,
                normalsBuffer = normalsBuffer,
                tangentsBuffer = tangentsBuffer,
                vectorBuffer = vectorBuffer,
                localToWorldMatrix = ren.localToWorldMatrix,
                basePosition = projectTarget.position,
                baseAxis = projectTarget.up,
                relativeDotBuffer = relativeDotBuffer
            };

            var maxOfIter = new int[] { pointsBuffer.Length, normalsBuffer.Length, tangentsBuffer.Length }.ToList().Max();

            jobWork.Schedule(maxOfIter, 100).Complete();

            pointsBuffer.CopyTo(points);
            normalsBuffer.CopyTo(normals);
            tangentsBuffer.CopyTo(tangents);

            mesh.vertices = points;
            mesh.normals = normals;
            mesh.tangents = tangents;

            float min = relativeDotBuffer.Min();
            int minIndex = relativeDotBuffer.IndexOf<float>(min);
            float max = relativeDotBuffer.Max();
            int maxIndex = relativeDotBuffer.IndexOf<float>(max);

            var dimension = max - min;
            var minPoint = projectTarget.position + min * projectTarget.up;
            var maxPoint = projectTarget.position + max * projectTarget.up;

            if (minTarget) minTarget.position = minPoint;
            if (maxTarget) maxTarget.position = maxPoint;
            if (minTarget && maxTarget)
            {
                var scale = Vector3.Distance(maxTarget.position, minTarget.position);
                projectTarget.localScale = new Vector3(0.2f, scale/2.0f, 0.2f);
                projectTarget.position = (minPoint + maxPoint) / 2.0f;
            }            

            pointsBuffer.Dispose();
            normalsBuffer.Dispose();
            tangentsBuffer.Dispose();
            vectorBuffer.Dispose();
            relativeDotBuffer.Dispose();
        }
        else
        {
            for (int i = 0; i < points.Length; i++) 
            {
                points[i] = ren.localToWorldMatrix * new Vector4(filter.sharedMesh.vertices[i].x, filter.sharedMesh.vertices[i].y, filter.sharedMesh.vertices[i].z, 1.0f);
                normals[i] = ren.localToWorldMatrix * filter.sharedMesh.normals[i];
                tangents[i] = ren.localToWorldMatrix * filter.sharedMesh.tangents[i];
            }
        }
        mesh.vertices = points;
    }
    private void OnDrawGizmos()
    {        
        Gizmos.color = wireColor;
        Gizmos.DrawWireMesh(mesh);
    }

}
[BurstCompile]
public struct JobWork : IJobParallelFor
{
    public NativeArray<Vector3> pointsBuffer;
    public NativeArray<Vector3> normalsBuffer;
    public NativeArray<Vector4> tangentsBuffer;
    public NativeArray<Vector3> vectorBuffer;
    public NativeArray<float> relativeDotBuffer;
    public Matrix4x4 localToWorldMatrix;
    public Vector3 basePosition;
    public Vector3 baseAxis;
    public void Execute(int index)
    {
        if (index < normalsBuffer.Length)
            normalsBuffer[index] = localToWorldMatrix * normalsBuffer[index];
        if (index < tangentsBuffer.Length)
            tangentsBuffer[index] = localToWorldMatrix * tangentsBuffer[index];

        if (index < pointsBuffer.Length)
        {
            pointsBuffer[index] = localToWorldMatrix * new Vector4(pointsBuffer[index].x, pointsBuffer[index].y, pointsBuffer[index].z, 1.0f);
            vectorBuffer[index] = pointsBuffer[index] - basePosition;
            relativeDotBuffer[index] = Vector3.Dot(vectorBuffer[index], baseAxis);

        }

    }
}
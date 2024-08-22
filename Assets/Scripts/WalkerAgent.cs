using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Linq;

public class WalkerAgent : Agent
{
    [Header("Body Parts")]
    public Transform hips;
    public Transform chest;
    public Transform spine;
    public Transform head;
    public Transform thighL;
    public Transform shinL;
    public Transform footL;
    public Transform thighR;
    public Transform shinR;
    public Transform footR;
    public Transform armL;
    public Transform forearmL;
    public Transform handL;
    public Transform armR;
    public Transform forearmR;
    public Transform handR;

    [Header("Torque Settings")]
    public float maxTorqueMagnitude = 100f;  // Maximum magnitude of the torque

    [Header("Spring Settings")]
    public float maxSpringValue = 1000f; // Maximum spring value for joints

    private Rigidbody[] bodyParts;
    private ConfigurableJoint[] joints;

    private Vector3[] initialPositions;
    private Quaternion[] initialRotations;

    // Action space sizes
    public int torqueActionSize = 48;  // 16 body parts * 3 torque directions
    public int springActionSize = 16;  // One for each of 16 joints

    // Initialize the agent
    public override void Initialize()
    {
        InitializeBodyParts();
        InitializeJoints();
        SaveInitialState();
    }

    private void InitializeBodyParts()
    {
        bodyParts = new Rigidbody[]
        {
            hips.GetComponent<Rigidbody>(), chest.GetComponent<Rigidbody>(), spine.GetComponent<Rigidbody>(), head.GetComponent<Rigidbody>(),
            thighL.GetComponent<Rigidbody>(), shinL.GetComponent<Rigidbody>(), footL.GetComponent<Rigidbody>(),
            thighR.GetComponent<Rigidbody>(), shinR.GetComponent<Rigidbody>(), footR.GetComponent<Rigidbody>(),
            armL.GetComponent<Rigidbody>(), forearmL.GetComponent<Rigidbody>(), handL.GetComponent<Rigidbody>(),
            armR.GetComponent<Rigidbody>(), forearmR.GetComponent<Rigidbody>(), handR.GetComponent<Rigidbody>()
        };
    }

    private void InitializeJoints()
    {
        joints = new ConfigurableJoint[]
        {
            hips.GetComponent<ConfigurableJoint>(), chest.GetComponent<ConfigurableJoint>(), spine.GetComponent<ConfigurableJoint>(), head.GetComponent<ConfigurableJoint>(),
            thighL.GetComponent<ConfigurableJoint>(), shinL.GetComponent<ConfigurableJoint>(), footL.GetComponent<ConfigurableJoint>(),
            thighR.GetComponent<ConfigurableJoint>(), shinR.GetComponent<ConfigurableJoint>(), footR.GetComponent<ConfigurableJoint>(),
            armL.GetComponent<ConfigurableJoint>(), forearmL.GetComponent<ConfigurableJoint>(), handL.GetComponent<ConfigurableJoint>(),
            armR.GetComponent<ConfigurableJoint>(), forearmR.GetComponent<ConfigurableJoint>(), handR.GetComponent<ConfigurableJoint>()
        };
    }

    private void SaveInitialState()
    {
        initialPositions = new Vector3[bodyParts.Length];
        initialRotations = new Quaternion[bodyParts.Length];

        for (int i = 0; i < bodyParts.Length; i++)
        {
            if (bodyParts[i] != null)
            {
                initialPositions[i] = bodyParts[i].transform.localPosition;
                initialRotations[i] = bodyParts[i].transform.localRotation;
            }
        }
    }

    // Collect observations to feed to the neural network
    public override void CollectObservations(VectorSensor sensor)
    {
        CollectBodyPartObservations(sensor);
        CollectJointObservations(sensor);
    }

    private void CollectBodyPartObservations(VectorSensor sensor)
    {
        foreach (var part in bodyParts)
        {
            if (part != null)
            {
                sensor.AddObservation(part.transform.localPosition);
                sensor.AddObservation(part.transform.localRotation);
                sensor.AddObservation(part.velocity);
                sensor.AddObservation(part.angularVelocity);
            }
        }
    }

    private void CollectJointObservations(VectorSensor sensor)
    {
        foreach (var joint in joints)
        {
            if (joint != null)
            {
                sensor.AddObservation(joint.slerpDrive.positionSpring / maxSpringValue); // Normalized spring value
            }
        }
    }

    // Apply torque and update joint spring values based on actions received
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        float[] torqueActions = actionBuffers.ContinuousActions.Array.Take(torqueActionSize).ToArray();
        float[] springActions = actionBuffers.ContinuousActions.Array.Skip(torqueActionSize).Take(springActionSize).ToArray();

        ApplyTorques(torqueActions);
        UpdateJointSprings(springActions);

        RewardForUprightPosition();
    }

    private void ApplyTorques(float[] torqueActions)
    {
        ApplyTorqueToPart(hips, torqueActions[0], torqueActions[1], torqueActions[2]);
        ApplyTorqueToPart(chest, torqueActions[3], torqueActions[4], torqueActions[5]);
        ApplyTorqueToPart(spine, torqueActions[6], torqueActions[7], torqueActions[8]);
        ApplyTorqueToPart(head, torqueActions[9], torqueActions[10], torqueActions[11]);

        ApplyTorqueToPart(thighL, torqueActions[12], torqueActions[13], torqueActions[14]);
        ApplyTorqueToPart(shinL, torqueActions[15], torqueActions[16], torqueActions[17]);
        ApplyTorqueToPart(footL, torqueActions[18], torqueActions[19], torqueActions[20]);

        ApplyTorqueToPart(thighR, torqueActions[21], torqueActions[22], torqueActions[23]);
        ApplyTorqueToPart(shinR, torqueActions[24], torqueActions[25], torqueActions[26]);
        ApplyTorqueToPart(footR, torqueActions[27], torqueActions[28], torqueActions[29]);

        ApplyTorqueToPart(armL, torqueActions[30], torqueActions[31], torqueActions[32]);
        ApplyTorqueToPart(forearmL, torqueActions[33], torqueActions[34], torqueActions[35]);
        ApplyTorqueToPart(handL, torqueActions[36], torqueActions[37], torqueActions[38]);

        ApplyTorqueToPart(armR, torqueActions[39], torqueActions[40], torqueActions[41]);
        ApplyTorqueToPart(forearmR, torqueActions[42], torqueActions[43], torqueActions[44]);
        ApplyTorqueToPart(handR, torqueActions[45], torqueActions[46], torqueActions[47]);
    }

    private void UpdateJointSprings(float[] springActions)
    {
        for (int i = 0; i < joints.Length; i++)
        {
            if (joints[i] != null)
            {
                JointDrive drive = joints[i].slerpDrive;
                drive.positionSpring = Mathf.Clamp(springActions[i] * maxSpringValue, 0, maxSpringValue);
                joints[i].slerpDrive = drive;
            }
        }
    }

    private void RewardForUprightPosition()
    {
        AddReward(Vector3.Dot(hips.up, Vector3.up));  // Reward for staying upright
        AddReward(Vector3.Dot(head.up, Vector3.up));  // Reward for keeping head upright
    }

    // Reset the agent's position and environment at the start of each episode
    public override void OnEpisodeBegin()
    {
        ResetAgent();
    }

    private void ResetAgent()
    {
        for (int i = 0; i < bodyParts.Length; i++)
        {
            if (bodyParts[i] != null)
            {
                bodyParts[i].velocity = Vector3.zero;
                bodyParts[i].angularVelocity = Vector3.zero;
                bodyParts[i].transform.localPosition = initialPositions[i];
                bodyParts[i].transform.localRotation = initialRotations[i];
            }
        }
    }

    // Apply torque to a specific body part based on actions
    void ApplyTorqueToPart(Transform part, float torqueX, float torqueY, float torqueZ)
    {
        if (part != null)
        {
            Rigidbody rb = part.GetComponent<Rigidbody>();
            if (rb != null)
            {
                Vector3 torque = new Vector3(torqueX, torqueY, torqueZ) * maxTorqueMagnitude;
                rb.AddTorque(torque);
            }
        }
    }

    // Optional: Manual control for testing purposes
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;

        // Example manual control for testing (optional)
        continuousActions[0] = Input.GetAxis("Horizontal");
        continuousActions[1] = Input.GetAxis("Vertical");
        // Continue adding manual control as needed for testing
    }
}

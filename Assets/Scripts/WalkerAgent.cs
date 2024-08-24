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
        CollectFootObservations(sensor);
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

    private void CollectFootObservations(VectorSensor sensor)
    {
        RaycastHit hit;
        foreach (var foot in new Transform[] { footL, footR })
        {
            if (Physics.Raycast(foot.position, Vector3.down, out hit, 1f))
            {
                sensor.AddObservation(hit.distance); // Distance to ground
            }
            else
            {
                sensor.AddObservation(1f); // No ground detected nearby
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

        RewardForLocomotion();
    }

    private void ApplyTorques(float[] torqueActions)
    {
        // Clamp the torque magnitude for stability
        float clampedTorque = Mathf.Clamp(maxTorqueMagnitude * 0.5f, 0, maxTorqueMagnitude);

        ApplyTorqueToPart(hips, torqueActions[0] * clampedTorque, torqueActions[1] * clampedTorque, torqueActions[2] * clampedTorque);
        ApplyTorqueToPart(chest, torqueActions[3] * clampedTorque, torqueActions[4] * clampedTorque, torqueActions[5] * clampedTorque);
        ApplyTorqueToPart(spine, torqueActions[6] * clampedTorque, torqueActions[7] * clampedTorque, torqueActions[8] * clampedTorque);
        ApplyTorqueToPart(head, torqueActions[9] * clampedTorque, torqueActions[10] * clampedTorque, torqueActions[11] * clampedTorque);

        ApplyTorqueToPart(thighL, torqueActions[12] * clampedTorque, torqueActions[13] * clampedTorque, torqueActions[14] * clampedTorque);
        ApplyTorqueToPart(shinL, torqueActions[15] * clampedTorque, torqueActions[16] * clampedTorque, torqueActions[17] * clampedTorque);
        ApplyTorqueToPart(footL, torqueActions[18] * clampedTorque, torqueActions[19] * clampedTorque, torqueActions[20] * clampedTorque);

        ApplyTorqueToPart(thighR, torqueActions[21] * clampedTorque, torqueActions[22] * clampedTorque, torqueActions[23] * clampedTorque);
        ApplyTorqueToPart(shinR, torqueActions[24] * clampedTorque, torqueActions[25] * clampedTorque, torqueActions[26] * clampedTorque);
        ApplyTorqueToPart(footR, torqueActions[27] * clampedTorque, torqueActions[28] * clampedTorque, torqueActions[29] * clampedTorque);

        ApplyTorqueToPart(armL, torqueActions[30] * clampedTorque, torqueActions[31] * clampedTorque, torqueActions[32] * clampedTorque);
        ApplyTorqueToPart(forearmL, torqueActions[33] * clampedTorque, torqueActions[34] * clampedTorque, torqueActions[35] * clampedTorque);
        ApplyTorqueToPart(handL, torqueActions[36] * clampedTorque, torqueActions[37] * clampedTorque, torqueActions[38] * clampedTorque);

        ApplyTorqueToPart(armR, torqueActions[39] * clampedTorque, torqueActions[40] * clampedTorque, torqueActions[41] * clampedTorque);
        ApplyTorqueToPart(forearmR, torqueActions[42] * clampedTorque, torqueActions[43] * clampedTorque, torqueActions[44] * clampedTorque);
        ApplyTorqueToPart(handR, torqueActions[45] * clampedTorque, torqueActions[46] * clampedTorque, torqueActions[47] * clampedTorque);
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

    private void RewardForLocomotion()
    {
        // Reward for moving forward based on the velocity of the hips
        Vector3 localVelocity = hips.InverseTransformDirection(hips.GetComponent<Rigidbody>().velocity);
        AddReward(localVelocity.z); // Reward for forward motion (along the z-axis)

        // Encourage staying upright
        AddReward(Vector3.Dot(hips.up, Vector3.up));  // Upright torso
        AddReward(Vector3.Dot(head.up, Vector3.up));  // Upright head

        // Encourage leg alignment and stability
        AddReward(Vector3.Dot(thighL.up, Vector3.up));  // Upright left thigh
        AddReward(Vector3.Dot(thighR.up, Vector3.up));  // Upright right thigh
        AddReward(Vector3.Dot(shinL.up, Vector3.up));  // Upright left shin
        AddReward(Vector3.Dot(shinR.up, Vector3.up));  // Upright right shin

        // Penalize for excessive rotation or falling
        if (hips.localPosition.y < -1f || head.localPosition.y < 1.3f)  // Example threshold for falling
        {
            SetReward(-1f);  // Large penalty for falling down
            EndEpisode();
        }
    }

    // Reset the agent to the initial state when a new episode starts
    public override void OnEpisodeBegin()
    {
        ResetAgentPosition();
    }

    private void ResetAgentPosition()
    {
        for (int i = 0; i < bodyParts.Length; i++)
        {
            if (bodyParts[i] != null)
            {
                bodyParts[i].transform.localPosition = initialPositions[i];
                bodyParts[i].transform.localRotation = initialRotations[i];
                bodyParts[i].velocity = Vector3.zero;
                bodyParts[i].angularVelocity = Vector3.zero;
            }
        }
    }

    private void ApplyTorqueToPart(Transform bodyPart, float xTorque, float yTorque, float zTorque)
    {
        if (bodyPart != null)
        {
            bodyPart.GetComponent<Rigidbody>().AddTorque(new Vector3(xTorque, yTorque, zTorque));
        }
    }

    // Use manual control (optional, for debugging or specific scenarios)
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;
        // Example of manual control, where each axis is controlled by input
        continuousActions[0] = Input.GetAxis("Horizontal");
        continuousActions[1] = Input.GetAxis("Vertical");
    }
}

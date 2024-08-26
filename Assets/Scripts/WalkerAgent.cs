using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Linq;
using System;

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

    // Action space sizes (increased by 2 for forward and upward movement)
    public int torqueActionSize = 48;  // 16 body parts * 3 torque directions
    public int springActionSize = 16;  // One for each of 16 joints
    public int movementActionSize = 2;  // Additional actions for forward and upward movement

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
        float forwardMovementAction = actionBuffers.ContinuousActions.Array[torqueActionSize + springActionSize];
        float upwardMovementAction = actionBuffers.ContinuousActions.Array[torqueActionSize + springActionSize + 1];

        ApplyTorques(torqueActions);
        UpdateJointSprings(springActions);

        // Apply forward and upward forces based on the agent's decision
        ApplyMovement(forwardMovementAction, upwardMovementAction);

        RewardForLocomotion();
    }

    private void ApplyTorques(float[] torqueActions)
    {
        // Clamp the torque magnitude for stability
        float clampedTorque = Mathf.Clamp(maxTorqueMagnitude * 0.75f, 0, maxTorqueMagnitude);

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

    // Reward shaping to encourage stepping and lifting one leg at a time
    private void RewardForLocomotion()
    {
        // Reward for forward velocity
        Vector3 velocity = hips.GetComponent<Rigidbody>().velocity;
        float forwardVelocity = -(velocity.z);
        AddReward(forwardVelocity * 0.1f); // Reward for moving forward

        // Penalize for staying stationary
        float totalVelocity = velocity.magnitude;
        if (totalVelocity < 0.1f)  // Threshold for being "stationary"
        {
            AddReward(-0.01f);  // Small penalty for being stationary
        }

        // Penalize for low hip height
        if (hips.position.y < 2.5f)
        {
            SetReward(-1f);  // Large penalty for falling down
            EndEpisode();
        }

        // Penalize for excessive rotation or falling
        if (head.position.y < 4.5f)  // Example threshold for falling
        {
            SetReward(-1f);  // Large penalty for falling down
            EndEpisode();
        }

        // Check if the agent is alternating foot steps by rewarding when one foot lifts after the other
        float footLHeight = footL.position.y;
        float footRHeight = footR.position.y;
        if (footLHeight > 0.7f && footRHeight < 0.7f)
        {
            AddReward(0.01f);  // Reward for lifting the left foot after the right foot was down
        }
        else if (footRHeight > 0.7f && footLHeight < 0.7f)
        {
            AddReward(0.01f);  // Reward for lifting the right foot after the left foot was down
        }
    }

    // Apply forward and upward forces based on the action values
    private void ApplyMovement(float forwardAction, float upwardAction)
    {
        Rigidbody hipsRb = hips.GetComponent<Rigidbody>();
        if (hipsRb != null)
        {
            // Apply forward force (scale based on the action value)
            Vector3 forwardForce = transform.forward * Mathf.Clamp(forwardAction, -1f, 1f) * 10f; // Adjust force scaling as needed
            hipsRb.AddForce(forwardForce, ForceMode.Acceleration);

            // Calculate the upward force multiplier based on the hips' Y position
            float hipsYPosition = hipsRb.position.y;
            float yThreshold = 3f; // Adjust this threshold based on your environment's needs
            float upwardForceMultiplier = 10f; // Default upward force multiplier

            // Increase the upward force multiplier if the hips are below the threshold
            if (hipsYPosition < yThreshold)
            {
                upwardForceMultiplier = Mathf.Lerp(30f, 100f, (yThreshold - hipsYPosition) / yThreshold); // Adjust the lerp range as needed
            }

            // Apply upward force (scale based on the action value and multiplier)
            Vector3 upwardForce = Vector3.up * Mathf.Clamp(upwardAction, 0f, 1f) * upwardForceMultiplier;
            hipsRb.AddForce(upwardForce, ForceMode.Acceleration);
        }
    }


    // Reset the agent's position, rotation, and velocity after each episode
    public override void OnEpisodeBegin()
    {
        for (int i = 0; i < bodyParts.Length; i++)
        {
            if (bodyParts[i] != null)
            {
                // Reset body parts to their initial positions and rotations
                bodyParts[i].transform.localPosition = initialPositions[i];
                bodyParts[i].transform.localRotation = initialRotations[i];
                bodyParts[i].velocity = Vector3.zero;
                bodyParts[i].angularVelocity = Vector3.zero;
            }
        }

        // Assign random rotations to the thighs for varied starting rotations
        float randomThighLRotation = UnityEngine.Random.Range(-90f, 30f);  // Random upward/downward rotation for left thigh
        float randomThighRRotation = UnityEngine.Random.Range(-90f, 30f);  // Random upward/downward rotation for right thigh

        // Apply the random rotations to the thighs
        //thighL.transform.localRotation = Quaternion.Euler(randomThighLRotation, thighL.transform.localRotation.eulerAngles.y, thighL.transform.localRotation.eulerAngles.z);
        //thighR.transform.localRotation = Quaternion.Euler(randomThighRRotation, thighR.transform.localRotation.eulerAngles.y, thighR.transform.localRotation.eulerAngles.z);

        // Assign random rotations to the shins (legs) for varied starting rotations
        float randomShinLRotation = UnityEngine.Random.Range(0f, 90f);  // Random forward/backward rotation for left shin
        float randomShinRRotation = UnityEngine.Random.Range(0f, 90f);  // Random forward/backward rotation for right shin

        // Apply the random rotations to the legs (shins)
        //shinL.transform.localRotation = Quaternion.Euler(randomShinLRotation, shinL.transform.localRotation.eulerAngles.y, shinL.transform.localRotation.eulerAngles.z);
        //shinR.transform.localRotation = Quaternion.Euler(randomShinRRotation, shinR.transform.localRotation.eulerAngles.y, shinR.transform.localRotation.eulerAngles.z);

    }

    // Apply torque to a body part
    private void ApplyTorqueToPart(Transform part, float torqueX, float torqueY, float torqueZ)
    {
        if (part != null && part.GetComponent<Rigidbody>() != null)
        {
            Rigidbody rb = part.GetComponent<Rigidbody>();
            rb.AddTorque(new Vector3(torqueX, torqueY, torqueZ));
        }
    }
}

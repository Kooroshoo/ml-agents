using UnityEngine;

public class WalkerAgent : MonoBehaviour
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

    [Header("Force and Torque Settings")]
    public Vector3 forceDirection = Vector3.up;  // Default direction for force
    public float forceMagnitude = 10f;           // Magnitude of the force
    public Vector3 torqueDirection = Vector3.forward; // Default direction for torque
    public float torqueMagnitude = 10f;          // Magnitude of the torque

    public bool applyForce = true;   // Toggle to apply force
    public bool applyTorque = false; // Toggle to apply torque
    public float testDuration = 3f;  // Duration to apply the force/torque
    private float elapsedTime;

    void Start()
    {
        elapsedTime = 0f;
    }

    void FixedUpdate()
    {
        if (elapsedTime < testDuration)
        {
            if (applyForce)
            {
                ApplyForceToAllParts();
            }

            if (applyTorque)
            {
                ApplyTorqueToAllParts();
            }

            elapsedTime += Time.fixedDeltaTime;
        }
    }

    // Apply force to all body parts with Rigidbody
    void ApplyForceToAllParts()
    {
        ApplyForceToPart(hips);
        ApplyForceToPart(chest);
        ApplyForceToPart(spine);
        ApplyForceToPart(head);
        ApplyForceToPart(thighL);
        ApplyForceToPart(shinL);
        ApplyForceToPart(footL);
        ApplyForceToPart(thighR);
        ApplyForceToPart(shinR);
        ApplyForceToPart(footR);
        ApplyForceToPart(armL);
        ApplyForceToPart(forearmL);
        ApplyForceToPart(handL);
        ApplyForceToPart(armR);
        ApplyForceToPart(forearmR);
        ApplyForceToPart(handR);
    }

    // Apply torque to all body parts with Rigidbody
    void ApplyTorqueToAllParts()
    {
        ApplyTorqueToPart(hips);
        ApplyTorqueToPart(chest);
        ApplyTorqueToPart(spine);
        ApplyTorqueToPart(head);
        ApplyTorqueToPart(thighL);
        ApplyTorqueToPart(shinL);
        ApplyTorqueToPart(footL);
        ApplyTorqueToPart(thighR);
        ApplyTorqueToPart(shinR);
        ApplyTorqueToPart(footR);
        ApplyTorqueToPart(armL);
        ApplyTorqueToPart(forearmL);
        ApplyTorqueToPart(handL);
        ApplyTorqueToPart(armR);
        ApplyTorqueToPart(forearmR);
        ApplyTorqueToPart(handR);
    }

    // Helper function to apply force to a specific part
    void ApplyForceToPart(Transform part)
    {
        if (part != null)
        {
            Rigidbody rb = part.GetComponent<Rigidbody>();
            if (rb != null)
            {
                rb.AddForce(forceDirection.normalized * forceMagnitude);
            }
        }
    }

    // Helper function to apply torque to a specific part
    void ApplyTorqueToPart(Transform part)
    {
        if (part != null)
        {
            Rigidbody rb = part.GetComponent<Rigidbody>();
            if (rb != null)
            {
                rb.AddTorque(torqueDirection.normalized * torqueMagnitude);
            }
        }
    }

    // Optional: Reset the test to run again
    public void ResetTest()
    {
        elapsedTime = 0f;
    }
}

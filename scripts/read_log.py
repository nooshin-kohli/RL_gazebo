import sys
import lcm

from obslcm import observ_t

import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    sys.stderr.write("usage: read-log <logfile>\n")
    sys.exit(1)
print(sys.argv[1])
log = lcm.EventLog(sys.argv[1], "r")
base_lin = []
base_ang = []
quat = []
q = []
qdot = []
tau = []
for event in log:
    if event.channel == "OBSERVATION":
        msg = observ_t.decode(event.data)
        base_lin.append(msg.base_lin_vel)
        base_ang.append(msg.base_ang_vel)
        quat.append(msg.quaternion)
        q.append(msg.dof_pos)
        qdot.append(msg.dof_vel)
        tau.append(msg.action)
        # print("response:::::",event.timestamp/1000)

        #print("Message:")
        #print("   timestamp   = %s" % str(msg.timestamp))
        #print("   position    = %s" % str(msg.position))
        #print("   orientation = %s" % str(msg.orientation))
        #print("   ranges: %s" % str(msg.ranges))
        #print("")

plt.figure()
plt.title("Base Lin Vel")
plt.plot(base_lin)


plt.figure()
plt.title("Base Ang Vel")
plt.plot(base_ang)


plt.figure()
plt.title("Quaternion")
plt.plot(quat)

plt.figure()
plt.title("DOF Pos")
plt.plot(q)


plt.figure()
plt.title("DOF Vel")
plt.plot(qdot)

plt.figure()
plt.title("DOF Tau")
plt.plot(tau)

plt.show()
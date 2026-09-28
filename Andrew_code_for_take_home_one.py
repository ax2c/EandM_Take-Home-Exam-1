#Andrew code for take home 1 
#I did not mark this at each individual point it was uesd, but I used gemini to help me with the actual plotting thats done here. 
#cs 121 did NOT prepare us for making graphs </3

####
# Plot the vector field of a line charge in the
# x-z plane, with the charge on the x-axis, straddling the origin.
# Eric Leibensperger - September 22, 2020
####
# Import necessary libraries/packages
import numpy as np
from matplotlib import pyplot as plt

# Create coordinates in x, z directions that go from
# -1.5 --> 1.5 and have 31 points
x = np.linspace(-1.5, 1.5, 31)
z = np.linspace(-1.5, 1.5, 31)

# Create 2D mesh
xMesh, zMesh = np.meshgrid(x, z)
# Save some typing by making X - 1/2 and X + 1/2 variables.
# Here the 1/2 is L/2 from the formula. We've scaled
# everything so L = 1
xMm = (xMesh) - 1/2
xMp = (xMesh) + 1/2

# Calculate the x-component
# Note that using the function hypot to calculate is more efficient
# than
# np.sqrt(zMesh**2+xMm**2)
Ex = 1/np.hypot(zMesh, xMm) - 1/np.hypot(zMesh, xMp)
# Calculate the z-component
Ez = -xMm/(zMesh*np.hypot(zMesh, xMm)) + (xMp)/(zMesh*np.hypot(zMesh, xMp))

# Ez is not defined when z = 0. So let's just set that Ez part to 0,
# if
# is a z = 0. And Ex and Ez should both be 0 (non-existant) inside of
# the
# line
Ez[np.where(z==0), :] = 0.


# Because the line is scaled to L = 1 centered at 0, its physical span is |x| <= 0.5 (not 1.0).
# Also, setting inside the wire should apply to both Ex and Ez on the 2D mesh.

#Since the line is scaled to L=1 centered at 0, its span is gonna be |x| <= 0.5 (and setting in the wire hits ex and ez)

wire_mask = (zMesh == 0) & (np.abs(xMesh) <= 0.5)  # find grid points on line plot
Ex[wire_mask] = 0.0                                 # Efields inside conductor/charge gets zeroed out
Ez[wire_mask] = 0.0

# Save some typing again....

# V(x, z) = ln( [ (0.5 - x) + sqrt((x - 0.5)^2 + z^2) ] / [ -(x + 0.5) + sqrt((x + 0.5)^2 + z^2) ] )
with np.errstate(divide='ignore', invalid='ignore'):
    num = (0.5 - xMesh) + np.hypot(xMesh - 0.5, zMesh)  #Integral upper limit evaluation
    den = -(xMesh + 0.5) + np.hypot(xMesh + 0.5, zMesh) #Integral lower limit evaluation
    V = np.log(num / den)                               #scalar potential matrix


# potential goes to infinity on wire. Sets max finite value for cleaner plotting
V[~np.isfinite(V)] = np.nanmax(V[np.isfinite(V)])

# Below is another version of plotting the result.
# Here we put the data on a log scale so that it makes
# the arrows look nicer in quiver. We use the sign() function
# to ensure that we keep correct direction.
ExS = np.sign(Ex) * np.log10(1 + abs(Ex))
EzS = np.sign(Ez) * np.log10(1 + abs(Ez))

# ==============================================================================
# FIGURE 1 (Question 3): Analytical E-field Overlaid on Potential Contours
# ==============================================================================
plt.figure(figsize=(8, 6.5))  # figure window for q3

#Overplot of electric potential as filled contours under quiver
# Choosing 25 contour levels between the 5th and 95th percentiles prevents singular spikes from blowing out colors
levels = np.linspace(np.percentile(V, 5), np.percentile(V, 95), 25)
c_plot = plt.contourf(xMesh, zMesh, V, levels=levels, cmap='viridis', extend='both')
cbar = plt.colorbar(c_plot)                          # colorbal for scalar potential
cbar.set_label('Electric Potential $V$ (arbitrary units)', fontsize=11)

# Handout plotting commands:
# make arrows white so you can see them
plt.quiver(xMesh, zMesh, ExS, EzS, color='white', scale=28, width=0.0035)

# Since L = 1 and straddles the origin, the endpoints are [-0.5, 0.5]. Line color changed to red for high visibility.
plt.plot([-0.5, 0.5], [0, 0], linewidth=5, color='red', label='Line charge ($L=1$)')
plt.xlabel('x', fontsize=12)
plt.ylabel('z', fontsize=12)
plt.title('Question 3: Analytical Electric Field & Potential Contours', fontsize=12)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# FIGURE 2 Q4: numerical E-field from -grad(V) overlaid on potential
# E = -grad(V) so Ex = -dV/dx,  Ez = -dV/dz.
# Opt 1 for spacing
dx = x[1] - x[0]
dz = z[1] - z[0]


grad_z, grad_x = np.gradient(V, dz, dx)
Ex_num = -grad_x
Ez_num = -grad_z

# zero out field
Ex_num[wire_mask] = 0.0
Ez_num[wire_mask] = 0.0

#symmetrical log scaling for numerical field componenets for display
ExS_num = np.sign(Ex_num) * np.log10(1 + abs(Ex_num))
EzS_num = np.sign(Ez_num) * np.log10(1 + abs(Ez_num))

plt.figure(figsize=(8, 6.5))  # Figure window for Q4
c_plot2 = plt.contourf(xMesh, zMesh, V, levels=levels, cmap='viridis', extend='both')
cbar2 = plt.colorbar(c_plot2)
cbar2.set_label('Electric Potential $V$ (arbitrary units)', fontsize=11)

plt.quiver(xMesh, zMesh, ExS_num, EzS_num, color='white', scale=28, width=0.0035)
plt.plot([-0.5, 0.5], [0, 0], linewidth=5, color='red', label='Line charge ($L=1$)')
plt.xlabel('x', fontsize=12)
plt.ylabel('z', fontsize=12)
plt.title(r'Question 4: Numerical Electric Field ($\vec{E} = -\nabla V$) & Potential Contours', fontsize=12)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

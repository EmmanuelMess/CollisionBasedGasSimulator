# Simple Explanation

This simulator computes all possible intersections, then checks which of those possible intersections is soonest to
occur, and checks if it is a real intersection that will happen, if it will, the simulator runs for Δt time, which
is enough for that collision to occur. Once the intersection occurs, the whole process runs again.
The idea is that CUDA allows this to occur very fast, by computing a lot of data in parallel (all segments and
intersections), and is very precise, not losing precision to particles teleporting.

### Some references

* [Colliding balls](https://garethrees.org/2009/02/17/physics/)

## Graph
      p1 .---------------->. p3   /
         |                 |      |Δd
       a .                 . b    /
         |                 |
      p2 .---------------->. p4

         /-----------------/
               ΔT * v0

a: Particle position at start of timestep  
b: Particle position at end of timestep  
Δd: Particle radius  
ΔT: maximum time step   
v0: fixed particle velocity  
p1 p3: Particle boundary during timestep  
p2 p4: Particle boundary during timestep  
a b: Particle path (only used for perpendiculars)  

## Collision calculation

### Between particles

Points ≈ $A$, $B$  
Velocities ≈ $A_v$, $B_v$  
Collsion time ≈ $t_I$  
Radius of particle ≈ $\Delta d$  
Distance between particles ≈ $d(X, Y)$  
Minimum representable value strictly greater than 0 ≈ $\delta$  

$A = (A_x, A_y)$   
$B = (B_x, B_y)$  
$Av = (A_{vx}, A_{vy})$  
$Bv = (B_{vx}, B_{vy})$  

$d(A + A_v t_I, B + B_v t_I) = 2 \Delta d$  
$d(A + A_v t_I, B + B_v t_I) = \sqrt{(A_x - B_x)^2 + (A_y - B_y)^2}$  

Then:

$d(A + A_v t, B + B_v t) = [(A_{vx} - B_{vx})^2 + (A_{vy} - B_{xy})^2] t^2 + 2 [(A_x - B_x)(A_{vx} - B_{vx}) + (A_y - B_y)(A_{vy} - B_{vy})] t + (A_x - B_x)^2 + (A_y-B_y)^2 - (2 \Delta d)^2$

Then:

$a = (A_{vx} - B_{vx})^2 + (A_{vy} - B_{xy})^2$  
$b = 2 [(A_x - B_x)(A_{vx} - B_{vx}) + (A_y - B_y)(A_{vy} - B_{vy})]$  
$c = (A_x - B_x)^2 + (A_y - B_y)^2 - (2 \Delta d)^2$  
$d = b^2 + 4 a c$  

And we use:

$$ t_1 = \max(\frac{-b - \sqrt{d}}{2 a},\sqrt{\delta}) $$

The $\sqrt{\delta}$ is used to prevent the simulation from freezing from a floating point underflow. The square root is
because it will get multiplied with a velocity that is $\sqrt{\delta}$ or larger (or 0), and velocity times timestep must
never be less than $\delta$.

#### Failure cases

* $d < 0$: No intersect  
* $b > -1e-6$: Glancing  
* $b >= 0$: Getting farther  
* $t0 < 0$ and $t1 > 0$ and $b <= -1e-6$: No intersect  

### Between particle and boundary

Point ≈ $A$  
Position (this is the same axis as wall, x for side collision, y for top/bottom) ≈ $A_p$  
Velocity (this is the same axis as wall, x for side collision, y for top/bottom) ≈ $A_v$  
Collision time ≈ $t_I$  
Radius of particle ≈ $\Delta d$   
Wall position (this is the X coordinate for the sides, and Y coordinate for top/bottom) ≈ $wall$   
Distance between particle and wall ≈ $d(X)$  
Minimum representable value strictly greater than 0 ≈ $\delta$

$d(A + A_v t_I, wall) = \Delta d$  
$d(A + A_v t_I, wall) = \sqrt{(A_p + A_v t_I - wall)^2}$

Then:

$d(A + A_v t_I, wall) = (A_p - wall)^2 - (\Delta d)^2 + 2 (A_p - wall) A_v + A_v^2 t_I^2$

Then:

$a = A_v^2$  
$b = 2 (A_p - wall) A_v$  
$c = (A_p - wall)^2 - (\Delta d)^2 = (A_p - wall + \Delta d) (A_p - wall - \Delta d)$  
$d = b^2 - 4 a c$  

And we use:

$$ t_1 = \max(\frac{-b - \sqrt{d}}{2 a}, \sqrt{\delta}) $$

The $\sqrt{\delta}$ is used to prevent the simulation from freezing from a floating point underflow. The square root is
because it will get multiplied with a velocity that is $\sqrt{\delta}$ or larger (or 0), and velocity times timestep must
never be less than $\delta$.

#### Failure cases

* $d < 0$: No intersect  
* $b > -1e-6$: Glancing  
* $b >= 0$: Getting farther  
* $t0 < 0$ and $t1 > 0$ and $b <= -1e-6$: No intersect


## Position calculation

Particle A position before ≈ $(x, y)$  
Particle A position after ≈ $(x', y')$  
Particle A velocity ≈ $(v_x, v_y)$  
timestep ≈ $\Delta t$  

$$(x', y') = (x + v_x \Delta t, y + v_y \Delta t)$$

## Velocity calculation

Velocity is only updated for colliding particles as there is no acceleration.

### Collision against wall
Particle position ≈ $(x, y)$  
Particle velocity before ≈ $(v_x, v_y)$  
Particle velocity after ≈ $({v'}_x, {v'}_y)$  

#### Vertical wall

$$({v'}_x, {v'}_y) = (-v_x, v_y)$$  

#### Horizontal wall

$$({v'}_x, {v'}_y) = (v_x, -v_y)$$  

### Collision particle

Particle A position ≈ $(A_x, A_y)$  
Particle A velocity before ≈ $(A_{vx}, A_{vy})$  
Particle A velocity after ≈ $(A \prime_{vx}, A \prime_{vy})$  
Particle B position ≈ $(B_x, B_y)$  
Particle B velocity before ≈ $(B_{vx}, B_{vy})$  
Particle B velocity after ≈ $(B \prime_{vx}, B \prime_{vy})$  
Minimum representable value strictly greater than 0 ≈ $\delta$

We model elastic collisions with particles of mass $1$ conserving momentum and kinetic energy (https://en.wikipedia.org/wiki/Elastic_collision):

$$(A \prime_{vx}, A \prime_{vy}) = (A_{vx}, A_{vy}) - \frac{(A_{vx} - B_{vx}) (A_x - B_x) + (A_{vy} - B_{vy}) (A_y - B_y)}{||(A_x, A_y) - (B_x, B_y)||^2} ((A_x, A_y) - (B_x, B_y))$$
$$(B \prime_{vx}, B \prime_{vy}) = (B_{vx}, B_{vy}) - \frac{(B_{vx} - A_{vx}) (B_x - A_x) + (B_{vy} - A_{vy}) (B_y - A_y)}{||(B_x, B_y) - (A_x, A_y)||^2} ((B_x, B_y) - (A_x, A_y))$$

The problem with this ecuation is that it accumulates floating point error due to all the floating subtractions.
To prevent this, we rewrite the ecuation into:

$$d = \frac{(A_{vx} - B_{vx}) (A_x - B_x) + (A_{vy} - B_{vy}) (A_y - B_y)}{||(A_x, A_y) - (B_x, B_y)||^2} ((A_x, A_y) - (B_x, B_y))$$  

$$(A \prime_{vx}, A \prime_{vy}) = (A_{vx}, A_{vy}) - d$$
$$(B \prime_{vx}, B \prime_{vy}) = (B_{vx}, B_{vy}) + d$$

To prevent a floating point underflow when the velocity is near zero and the delta is small we use a minimal velocity $\sqrt{\delta}$.
The square root is because it will get multiplied with a $\Delta t$ that is $\sqrt{\delta}$ or larger, and velocity 
times timestep must never be less than $\delta$.

If $(A_{vx}, A_{vy}) - d < \sqrt{\delta}$ then $(A \prime_{vx}, A \prime_{vy}) = 0$ otherwise $(A \prime_{vx}, A \prime_{vy}) = (A_{vx}, A_{vy}) - d$.  
If $(B_{vx}, B_{vy}) - d < \sqrt{\delta}$ then $(B \prime_{vx}, B \prime_{vy}) = 0$ otherwise $(B \prime_{vx}, B \prime_{vy}) = (B_{vx}, B_{vy}) + d$.

#### Floating point underflow reduction

Particle A velocity before collision ≈ $(A_{vx}, A_{vy})$  
Particle B velocity before collision ≈ $(B_{vx}, B_{vy})$  
$d$ update from previous section ≈ $d$

The inner product of the velocities must be equal before and after, any deviation is a floating point calculation error.

$$(A_{vx} -d) (B_{vx} + d) + (A_{vy} -d) (B_{vy} + d) = A_{vx} B_{vx} + A_{vy} B_{vy}$$  
$$A_{vx}B_{vx} + A_{vy}B_{vy} + d (A_{vx}-B_{vx}+A_{vy}-B_{vy}) - 2d^2 = A_{vx} B_{vx} + A_{vy} B_{vy}$$  
$$d (A_{vx}-B_{vx}+A_{vy}-B_{vy}) - 2d^2 = 0$$  

Now assume that $d_\varepsilon = d + \varepsilon$ is the incorrect $d$ value obtained by floating point operations:

$$(d+\varepsilon) (A_{vx}-B_{vx}+A_{vy}-B_{vy}) - 2(d+\varepsilon)^2 = 0$$  

The affected terms are:

$$E = \varepsilon (A_{vx}+A_{vy}) - \varepsilon(B_{vx}+B_{vy}) - 4d\varepsilon - 2\varepsilon^2$$

In the long run $(A_{vx}+A_{vy}) \approx (B_{vx}+B_{vy})$ so we don't care for the first two terms. What's left is 
$E \approx - 2\varepsilon^2 -4 d \varepsilon$ where $-4 d \varepsilon$ is neutral (when $d$ is positive it will be negative and 
vice versa), and $- 2\varepsilon^2$ which will make the simulation lose velocity in the long term.

## Communicating Sequential Processes model

#### Designations
all calls to cudaAlloc ≈ allocation  
graph optimization ≈ optimization  
setting particle position, initial velocities in device global memory ≈ initialization  
Move particles and cleanup sim ≈ advanceSimulation  
amount of particles ≈ PARTICLES  

#### CSP
CUDA = (||^(PARTICLES * PARTICLES)_i=1 i: i.computeIntersectionTime) -> (||^(PARTICLES)_i=1 i: i.calculateIntersectionBorderTime)  
          -> (||^(1)_i=1 i: i.minimum) -> advanceSimulation  
SIMULATION = CUDA -> SIMULATION  

DEVICE = CUDA -> DEVICE  
HOST = allocation -> initialization -> checkDistances -> optimization -> SIMULATION  
SYSTEM = HOST || DEVICE  

## Implementation details


### Saving collision time data

Particle A ≈ i  
Particle B ≈ j  
Collision time between A and a wall ≈ a  
Collision time between A and B ≈ b  
Data isn't saved because it would be duplicated ≈ x  

                    i
            0   1   2   3   4   5  
        0   a | b | b | b | b | b |
        1   x | a | b | b | b | b |
    j   2   x | x | a | b | b | b |
        3   x | x | x | a | b | b |
        4   x | x | x | x | a | b |
        5   x | x | x | x | x | a |
   

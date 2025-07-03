using LinearAlgebra

# --- Constants ---
G = 1.0  # Gravitational constant in normalized units

# --- Functions ---

"Compute accelerations and jerks for all bodies"
function compute_acc_jerk(pos, vel, mass)
    acc = [zeros(n) for _ in 1:n]
    jerk = [zeros(n) for _ in 1:n]

    for i in 1:n
        for j in 1:n
            if i != j
                r = pos[j] - pos[i]
                v = vel[j] - vel[i]
                dist2 = dot(r, r)
                dist3 = dist2 * sqrt(dist2)

                acc[i] += G * mass[j] * r / dist3
                jerk[i] += G * mass[j] * (v / dist3 - 3 * dot(r, v) * r / dist3 / dist2)
            end
        end
    end
    return acc, jerk
end

"Hermite integration step"
function hermite_step!(pos, vel, mass, dt)
    acc, jerk = compute_acc_jerk(pos, vel, mass)

    pos_pred = [p + v * dt + 0.5 * a * dt^2 + (1/6) * j * dt^3 for (p, v, a, j) in zip(pos, vel, acc, jerk)]
    vel_pred = [v + a * dt + 0.5 * j * dt^2 for (v, a, j) in zip(vel, acc, jerk)]

    acc_pred, jerk_pred = compute_acc_jerk(pos_pred, vel_pred, mass)

    pos .= [p + 0.5 * dt * (v + vp) + (1/12) * dt^2 * (a - ap) for (p, v, vp, a, ap) in zip(pos, vel, vel_pred, acc, acc_pred)]
    vel .= [v + 0.5 * dt * (a + ap) + (1/12) * dt * (j - jp) for (v, a, ap, j, jp) in zip(vel, acc, acc_pred, jerk, jerk_pred)]
end

"Total energy for diagnostic"
function total_energy(pos, vel, mass)
    kinetic = sum(0.5 * mass[i] * dot(vel[i], vel[i]) for i in 1:length(mass))
    potential = 0.0
    for i in 1:length(mass)-1
        for j in i+1:length(mass)
            r = pos[j] - pos[i]
            dist = norm(r) + 1e-10
            potential -= G * mass[i] * mass[j] / dist
        end
    end
    return kinetic + potential
end

# Get n, dt and t_max from command-line arguments
if length(ARGS) < 3
    println("Usage: julia script.jl <n> <dt> <t_max>")
    exit(1)
end

start_time = time()

# --- Initial Conditions ---
n = parse(Int, ARGS[1])
dt = parse(Float64, ARGS[2])
t_max = parse(Float64, ARGS[3])
integration_steps = Int(ceil(t_max / dt))
mass = [1.0 for _ in 1:n]

pos = zeros(integration_steps, n, 3)
vel = zeros(integration_steps, n, 3)
acc = zeros(integration_steps, n, 3)
jks = zeros(integration_steps, n, 3)

for i in 1:n
    phi = i * 2 * pi / 3
    pos[1][i][1] = cos(phi)
    pos[1][i][2] = sin(phi)
    pos[1][i][3] = 0
end

v_abs_init = 1.0 / sqrt(sqrt(3))
for i in 1:n
    phi = i * 2 * pi / 3
    vel[1][i][1] = - v_abs_init * sin(phi)
    vel[1][i][2] = v_abs_init * cos(phi)
    vel[1][i][3] = 0
end
vel[1][1] += 0.0001

# --- Integration Loop ---

n_steps = Int(t_max / dt)

initial_energy = total_energy(pos, vel, mass)
orbits = []
for step in 1:n_steps
    hermite_step!(pos, vel, mass, dt)
    push!(orbits, deepcopy(pos))
    # if step % 100 == 0
    #     println("Time: $(step*dt), Energy conservation: $(initial_energy - total_energy(pos, vel, mass))")
    # end
end
println("Steps integrated: $(length(orbits))")
final_energy = total_energy(pos, vel, mass)
println("Energy conservation: $(initial_energy - final_energy)")

end_time = time()
println("Elapsed integration time: ", end_time - start_time, " seconds")

using Plots

t_len = length(orbits)         # number of timesteps

plt = plot(aspect_ratio=:equal)  # start an empty plot
for i in 1:n
    # if (i * dt) % 0.01 != 0
    #     continue
    # end
    x = [orbits[t][i][1] for t in 1:t_len]  # x positions over time
    y = [orbits[t][i][2] for t in 1:t_len]  # y positions over time
    
    plot!(x, y, label="Object $i")  # add trajectory for this object
end

plot_end_time = time()
println("Elapsed plotting time: ", plot_end_time - end_time, " seconds")

plot!(xlabel="x", ylabel="y", title="Energy cons. frac. $((initial_energy - final_energy) / initial_energy)")
display(plt)
savefig(plt, "orbit_plot_$(dt)_$(t_max).png")  # saves the plot
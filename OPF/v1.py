import GridCalEngine.api as gce
from GridCalEngine.Simulations.OPF.NumericalMethods.ac_opf import run_nonlinear_opf, ac_optimal_power_flow
from GridCalEngine.enumerations import TransformerControlType, AcOpfMode, ReactivePowerControlMode

# Change Sbase accordingly
grid = gce.MultiCircuit('Carles 6-bus system', Sbase=100.0, fbase=50.0)

# Bus creation, change vnom accordingly
bus1 = gce.Bus(name='Bus 1', vnom=200.0, vmax=1.1, vmin=0.9, is_slack=False)
bus2 = gce.Bus(name='Bus 2', vnom=200.0, vmax=1.1, vmin=0.9, is_slack=False)
bus3 = gce.Bus(name='Bus 3', vnom=200.0, vmax=1.1, vmin=0.9, is_slack=False)
bus4 = gce.Bus(name='Bus 4', vnom=200.0, vmax=1.1, vmin=0.9, is_slack=False)
bus5 = gce.Bus(name='Bus 5', vnom=200.0, vmax=1.1, vmin=0.9, is_slack=False)
bus6 = gce.Bus(name='Bus 6', vnom=200.0, vmax=1.1, vmin=0.9, is_slack=True)

grid.add_bus(bus1)
grid.add_bus(bus2)
grid.add_bus(bus3)
grid.add_bus(bus4)
grid.add_bus(bus5)
grid.add_bus(bus6)

# Line creation, change r, x, b, and rate accordingly
line2_3 = gce.Line(name='Line 2-3', bus_from=bus2, bus_to=bus3, r=0.001, x=0.002, b=0.001, rate=1000.0)
line3_4 = gce.Line(name='Line 3-4', bus_from=bus3, bus_to=bus4, r=0.001, x=0.002, b=0.001, rate=1000.0)
line5_6 = gce.Line(name='Line 5-6', bus_from=bus5, bus_to=bus6, r=0.001, x=0.002, b=0.0, rate=1000.0)

grid.add_line(line2_3)
grid.add_line(line3_4)
grid.add_line(line5_6)

# Transformer creation, change r, x, and rate accordingly
tr1_2 = gce.Transformer2W(name='Transformer 1-2', bus_from=bus1, bus_to=bus2, r=0.001, x=0.002, rate=1000.0)
tr4_5 = gce.Transformer2W(name='Transformer 4-5', bus_from=bus4, bus_to=bus5, r=0.001, x=0.002, rate=1000.0)

grid.add_transformer2w(tr1_2)
grid.add_transformer2w(tr4_5)

# Wind farm creation, change P and Q accordingly
gen1 = gce.StaticGenerator(name='Wind Farm 1', P=5.0, Q=0.0, cost=0.0)

grid.add_static_generator(bus1, gen1)

# Grid slack creation, change vset accordingly
gen2 = gce.Generator(name='Grid Slack', vset=1.0, Snom=100000.0, Cost=1.0)

grid.add_generator(bus6, gen2)

# Reactive power compensation creation, change limits accordingly
react1 = gce.Generator(name='Shunt 1', Qmin=0.0, Qmax=500.0, Pmin=-1e-5, Pmax=1e-5, Cost=0.0)
react2 = gce.Generator(name='Shunt 2', Qmin=0.0, Qmax=500.0, Pmin=-1e-5, Pmax=1e-5, Cost=0.0)
react3 = gce.Generator(name='Shunt 3', Qmin=0.0, Qmax=500.0, Pmin=-1e-5, Pmax=1e-5, Cost=0.0)
react4 = gce.Generator(name='Shunt 4', Qmin=0.0, Qmax=500.0, Pmin=-1e-5, Pmax=1e-5, Cost=0.0)
react5 = gce.Generator(name='Shunt 5', Qmin=0.0, Qmax=500.0, Pmin=-1e-5, Pmax=1e-5, Cost=0.0)
 
grid.add_generator(bus1, react1)
grid.add_generator(bus2, react2)
grid.add_generator(bus3, react3)
grid.add_generator(bus4, react4)
grid.add_generator(bus5, react5)


# Run PF + OPF
options = gce.PowerFlowOptions(gce.SolverType.NR, verbose=False)
power_flow = gce.PowerFlowDriver(grid, options)
power_flow.run()

opf_options = gce.OptimalPowerFlowOptions(solver=gce.SolverType.NONLINEAR_OPF, acopf_mode=AcOpfMode.ACOPFstd,
                                              ips_tolerance=1e-6, ips_iterations=50, verbose=1)
res = run_nonlinear_opf(grid=grid, pf_options=options, opf_options=opf_options, plot_error=True, pf_init=True)
print('')

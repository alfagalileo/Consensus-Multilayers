using OrdinaryDiffEq, DiffEqCallbacks
using PyCall
using LinearAlgebra
using BlockArrays
using DelimitedFiles
using IterTools
using ProgressBars
using LaTeXStrings

# import the needed python libraries
nx = pyimport("networkx")
plt = pyimport("matplotlib.pyplot")
pyx = pyimport("pyx")


function pyplot_solution( sol::ODESolution )
	s_ = size( sol.u, 1 )
	n_ = size( sol.u[1], 1 )
	z = Array{Float64}( undef, s_ , n_ )

	for i in 1:s_
		z[i,:] = sol.u[i][:]
	end

	for si in eachcol( z[:,1:Int64(n_/2)] )
		plt.plot( sol.t, si )
	end

	plt.show()
end


function export_data( sol::ODESolution )
	s_ = size( sol.u, 1 )
	n_ = size( sol.u[1], 1 )
	z = Array{Float64}( undef, s_ , n_ )

	for i in 1:s_
		z[i,:] = sol.u[i][:]
	end

	writedlm( "data.csv", hcat( sol.t, z[:,1:Int64(n_/2)]  ), "," )
end


"""
  # The make_structure Function
  `make_structure()`: build the multilayer structure.
  This function use Networkx to build the selected graph
"""
function make_structure( nodes_per_layer::Vector{Int64}, intralayer_struct::Dict, 
		interlayer_struct::Dict, Dx::Dict; verbose = true, show = false )
 	@assert length(nodes_per_layer) == length(intralayer_struct)

 	nlayers = length(nodes_per_layer)
 	G = nx.MultiGraph()

 	if verbose println(">> Creating nodes ...") end
	
	# build all the nodes
 	for layeri in 1:nlayers
 		for ni in 1:nodes_per_layer[layeri]
 			if verbose @info "building node $(ni) in layer $(layeri)" end
 			G.add_node( "L$(layeri)_$(ni)", layer=layeri )
 		end
 	end

 	if verbose println(">> Creating intralayer edges ...") end
	
	# make edges from nodes living in the same layer
 	for layer_i in 1:nlayers
 		edges_layer_i = intralayer_struct[ layer_i, layer_i ]

 		for nodes_i in edges_layer_i
 			if verbose	@info "conecting L$(layer_i)_$(nodes_i[1]) -> L$(layer_i)_$(nodes_i[2])" end
 			G.add_edge( "L$(layer_i)_$(nodes_i[1])", "L$(layer_i)_$(nodes_i[2])" )
 		end
 	end

 	if verbose println(">> Creating interlayer edges ...") end
	
	# make edges from nodes interlayer with Dx amplitude
 	for pair_i in keys( interlayer_struct )
 		layer_from, layer_to = pair_i

 		for nodes_i in interlayer_struct[ pair_i... ]
 			if verbose @info "conecting L$(layer_from)_$(nodes_i[1]) -> L$(layer_to)_$(nodes_i[2])" end
 			G.add_edge( "L$(layer_from)_$(nodes_i[1])", "L$(layer_to)_$(nodes_i[2])", weight = Dx[layer_from, layer_to] )
 		end
 	end

 	if show
 		pos = nx.spring_layout(G)
 		nx.draw( G, pos, with_labels=true, edge_color = "gray", node_size = 2000)
 		plt.show()
 	end
 	
 	L = nx.laplacian_matrix(G).toarray()

 	# build the diagonal diffusion coefficients
 	_size = sum( nodes_per_layer )
 	D = BlockArray{Float64}( zeros( _size, _size ), nodes_per_layer, nodes_per_layer )

 	for i in 1:nlayers
 		D[Block(i,i)] = 0.0*Matrix{Float64}( I, nodes_per_layer[i], nodes_per_layer[i] )
 	end

 	return L + Array( D )
end


"""
Right hand side of evolution equation
"""
function RHS!( du, u, p, t )
	L, I, n = p 

	du[1:n] = u[n+1:end]
	du[n+1:end] = -( L +  I )*u[n+1:end] - L^2*u[1:n]
end


function condition( u, t, integrator ) 
	s = 0.0
	_n = Int64(length(u)/2)
	for (i,j) in subsets( 1:_n, 2 )
		s += abs(u[i] - u[j])
	end

	return s <= 1e-2
end


function solve_dynamics_ss( L )
	affect!( integrator ) = terminate!(integrator)
	cb = DiscreteCallback( condition, affect! )

	# solve the dynamics problem
	_n = size(L,1)
	p = ( L, Matrix( I, _n, _n ), _n )
	tspan = ( 0.0, 10_000.0 )
	u0 = vcat( 1.0:1.0:_n, zeros(_n) )
	
	prob = ODEProblem{true}( RHS!, u0, tspan, p )
	sol = solve( prob, DP5(), callback = cb  )
	return sol
end


function solve_dynamics( L; tf::Float64 = 10.0 )
	# solve the dynamics problem
	_n = size(L,1)
	p = ( L, Matrix( I, _n, _n ), _n )
	tspan = ( 0.0, tf )
	u0 = vcat( 1.0:1.0:_n, zeros(_n) )
	
	prob = ODEProblem{true}( RHS!, u0, tspan, p )
	sol = solve( prob, RK4() )
	return sol
end


function plot_solution( params::Dict, sol::ODESolution )
	s_ = size( sol.u, 1 )
	n_ = size( sol.u[1], 1 )
	z = Array{Float64}( undef, s_ , n_ )

	for i in 1:s_
		z[i,:] = sol.u[i][:]
	end

	# define layer's colors
	cols = []
	col_options = [ pyx.color.cmyk.Fuchsia, pyx.color.cmyk.BurntOrange, pyx.color.cmyk.BrickRed, pyx.color.cmyk.Green ]
	
	for ( i, sizei ) in enumerate( params[:nodes_per_layer] )
		append!( cols, repeat([ col_options[i] ], sizei ) )
	end

	c = pyx.canvas.canvas()

	g = c.insert(
		pyx.graph.graphxy(
			width = 7,
			x = pyx.graph.axis.lin( min = -20, max = 300, title = "Time (au)" ),
			y = pyx.graph.axis.lin( min = 0, max = 20, title = L"$y_k$"),
			#x = pyx.graph.axis.lin( min = -1, max = 15, title = "Time (au)" ),
			#y = pyx.graph.axis.lin( min = 0, max = 11, title = L"$y_k$"),
			#x = pyx.graph.axis.lin( min = -2, max = 40, title = "Time (au)" ),
			#y = pyx.graph.axis.lin( min = 0, max = 9, title = L"$y_k$"),
		)
	)

	for (i, si) in enumerate( eachcol( z[:,1:Int64(n_/2)] ) )
		g.plot(
			pyx.graph.data.values( x = sol.t, y = si ),
			[
				pyx.graph.style.line([ pyx.style.linewidth.thick, cols[i] ])
			]
		)
	end

	
	h = c.insert(
		pyx.graph.graphxy(
			xpos = 4,
			ypos = 3.0,
			width = 2.8,
			x = pyx.graph.axis.lin( min = 0, max = 5, title = "" ),
			y = pyx.graph.axis.lin( min = 0, max = 20, title = "" ),
			backgroundattrs = [ pyx.deco.filled([ pyx.color.rgb.white ]) ]
		)
	)

	for (i, si) in enumerate( eachcol( z[:,1:Int64(n_/2)] ) )
		h.plot(
			pyx.graph.data.values( x = sol.t, y = si ),
			[
				pyx.graph.style.line([ pyx.style.linewidth.thick, cols[i] ])
			]
		)
	end
	

	pos_ini = [ 200.0, 5.0 ]
	for ( i, sizei ) in enumerate( params[:nodes_per_layer] )
		g.fill(
			pyx.path.rect( collect( g.pos( pos_ini... ) )..., 0.5, 0.1 ),
			[
				col_options[i]
			]
		)

		g.text( collect( g.pos( pos_ini[1]+40.0, pos_ini[2] ) )..., L"$L_%$i$",
			[
				pyx.text.size.small
			]
		)

		pos_ini[:] = pos_ini + [0,-1.2]
	end

	c.writeSVGfile("plot.svg")
	c.writeEPSfile("plot.eps")
end



function meshgrid(x, y)
	X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end


function cost_landscape( sol::ODESolution )
	n = size(sol,1)
	@show n

	z = Array{Float64}( undef, size(sol,2) , size(sol,1) )
	
	for i in 1:length(sol)
		z[i,:] = sol.u[i][:]
	end

	plt.plot( z[:,3], z[:,7] )
	plt.show()
end


function main()
	#=
	# number of nodes in each layer (in the same order)
	nodes_per_layer = [5, 5]

	# structure of the nodes conections in each layer
	Edges_Intra = Dict()
	Edges_Intra[1,1] = [ [i, (i == nodes_per_layer[1]) ? 1 : i+1] for i in 1:nodes_per_layer[1] ]
	Edges_Intra[2,2] = [ [i,j] for (i,j) in IterTools.subsets( 1:nodes_per_layer[2], Val{2}() ) ]

	# structure of the conections of nodes among layers. For example L1_1 -> L2_2, L1_3 -> L2_4
	Edges_Inter = Dict()
	Edges_Inter[1,2] = [ [i,i] for i in 1:nodes_per_layer[1] ]

	Dx = Dict( i => 0.4 for i in keys(Edges_Inter) )

	# construct the laplacian matrix of the graph
	L = make_structure( nodes_per_layer, Edges_Intra, Edges_Inter, Dx )
	=#



	# number of nodes in each layer (in the same order)
	nodes_per_layer = [3, 5]

	# structure of the nodes conections in each layer
	Edges_Intra = Dict()
	Edges_Intra[1,1] = [ [i, (i== nodes_per_layer[1]) ? 1 : i+1] for i in 1:nodes_per_layer[1] ]
	Edges_Intra[2,2] = [ [i, (i== nodes_per_layer[2]) ? 1 : i+1] for i in 1:nodes_per_layer[2] ]

	# structure of the conections of nodes among layers. For example L1_1 -> L2_2, L1_3 -> L2_4
	Edges_Inter = Dict()
	Edges_Inter[1,2] = [ [1,2], [3,4] ]

	# define the interlayer difussion constants
	Dx = Dict( i => 0.5 for i in keys(Edges_Inter) )

	# construct the laplacian matrix of the graph
	L = make_structure( nodes_per_layer, Edges_Intra, Edges_Inter, Dx )


	#=
	# number of nodes in each layer (in the same order)
	nodes_per_layer = [ 3, 4, 5, 6 ]

	# structure of the nodes conections in each layer
	Edges_Intra = Dict()
	Edges_Intra[1,1] = [ [1,2], [2,3], [3,1] ]
	Edges_Intra[2,2] = [ [1,2], [2,4], [4,3], [3,1], [1,4] ]
	Edges_Intra[3,3] = [ [1,2], [2,3], [3,4], [4,5], [5,1], [1,4], [1,3] ]
	Edges_Intra[4,4] = [ [1,5], [5,3], [2,4], [2,6], [2,5] ]

	# structure of the conections of nodes among layers. For example L1_1 -> L2_2, L1_3 -> L2_4
	Edges_Inter = Dict()
	Edges_Inter[1,2] = [ [1,2] ]
	Edges_Inter[1,3] = [ [2,1] ]
	Edges_Inter[1,4] = [ [3,2] ]
	Edges_Inter[2,3] = [ [3,5] ]
	Edges_Inter[2,4] = [ [4,1] ]
	Edges_Inter[3,4] = [ [3,5] ]

	# define the interlayer difussion constants
	# Dx = Dict()
	# Dx[1,2] = 0.1
	# Dx[1,3] = 0.5
	# Dx[1,4] = 0.2
	# Dx[2,3] = 0.1
	# Dx[2,4] = 0.2
	# Dx[3,4] = 0.3


	Dx = Dict()
	Dx[1,2] = 1.0
	Dx[1,3] = 1.0
	Dx[1,4] = 1.0
	Dx[2,3] = 0.05
	Dx[2,4] = 0.05
	Dx[3,4] = 0.05

	# construct the laplacian matrix of the graph
	L = make_structure( nodes_per_layer, Edges_Intra, Edges_Inter, Dx )
	=#


	sol = solve_dynamics( L, tf = 300.0 )
	# sol = solve_dynamics_ss( L )
	# pyplot_solution( sol )
	# export_data( sol )
	cost_landscape( sol )
	
	# params = Dict( :nodes_per_layer => nodes_per_layer )
	# plot_solution( params, sol )


	#=
	dx = LinRange( 0.1, 1.5, 500 )
	Tc = Vector{Float64}( undef, length(dx) )
	
	Dx = Dict()
	for (i, di) in ProgressBar( enumerate(dx) )
		Dx[1,2] = 0.1
		Dx[1,3] = 0.5
		Dx[1,4] = 0.2
		Dx[1,4] = di
		Dx[2,3] = 0.1
		Dx[2,4] = 0.2
		Dx[3,4] = 0.3

		L = make_structure( nodes_per_layer, Edges_Intra, Edges_Inter, Dx, verbose = false, show = false )
		sol = solve_dynamics_ss( L  )
		Tc[i] = sol.t[end]
	end

	# writedlm("Dx[3,4].dat", hcat(dx, Tc) )
	plt.semilogy( dx, Tc )
	plt.show()
	=#
end


using OrdinaryDiffEq, DiffEqCallbacks
using PyCall, PyPlot
using LinearAlgebra
using BlockArrays
using DelimitedFiles
using IterTools

nx = pyimport("networkx")

function pyplot_solution( sol::ODESolution )
    s_ = size( sol.u, 1 )
    n_ = size( sol.u[1], 1 )
    z = Array{Float64}( undef, s_ , n_ )

    for i in 1:s_
      z[i,:] = sol.u[i][:]
    end

    for si in eachcol( z[:,1:Int64(n_/2)] )
      plot( sol.t, si )
    end
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
            if verbose      @info "conecting L$(layer_i)_$(nodes_i[1]) -> L$(layer_i)_$(nodes_i[2])" end
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

    return L
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
    sol = solve( prob, DP5(), callback = cb) # Removed callback here
    return sol
end

function solve_dynamics( L; tf::Float64 = 10.0 )
    # solve the dynamics problem
    _n = size(L,1)
    p = ( L, Matrix( I, _n, _n ), _n )
    tspan = ( 0.0, tf )
    u0 = vcat( 1.0:1.0:_n, zeros(_n) )

    prob = ODEProblem{true}( RHS!, u0, tspan, p )
    sol = solve( prob, DP5()  )
    return sol
end


function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end


function main( ;which = [1,2], fixed_val = 0.2, num=20 )
    # which -> vector de dos dimensiones para elegir cual de las constantes mover
    # En este caso los indices no pueden superar 6
    # fixed_val -> valor de la constante para las demad Dx

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

    pairs_inter = collect( keys(Edges_Inter) )
    pairs_inter_subset = setdiff( pairs_inter, pairs_inter[which] )

    Dx = Dict()
    for idx in pairs_inter_subset
        Dx[idx] = fixed_val
    end

    dx = LinRange( 0.1, 1.0, num )
    DX,DY = meshgrid(dx,dx)

    Tc = Array{Float64}( undef, length(dx), length(dx) )

	for i in 1:length(dx)
        @info i
        for j in 1:length(dx)

            DDX = [ DX[i,j], DY[i,j] ]
            for ( k, idx ) in enumerate( pairs_inter[which] )
                Dx[idx] = DDX[k]
            end

            L = make_structure( nodes_per_layer, Edges_Intra, Edges_Inter, Dx, verbose = false, show = false )
            sol = solve_dynamics_ss( L )
            Tc[i,j] = sol.t[end]
        end
	end

	writedlm("Dx$(which)_fixed=$(fixed_val).dat", Tc )
    return Tc
end

#main( which = [1,2], fixed_val = 0.5, num = 20 )


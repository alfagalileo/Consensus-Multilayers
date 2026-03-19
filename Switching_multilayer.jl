
using LinearAlgebra
using SparseArrays
using OrdinaryDiffEq
using Random
using PyCall

plt = pyimport("matplotlib.pyplot")
pyx = pyimport("pyx")

Random.seed!(1234)

const N1 = 3
const N2 = 5
const T_max = 20.0
const Δt_switch = 2.5
const D = 0.4


function ring_laplacian(n)
    L = spzeros(n, n)
    for i in 1:n
        L[i, mod1(i-1, n)] = -1
        L[i, mod1(i+1, n)] = -1
        L[i, i] = 2
    end
    return L
end


function random_laplacian(n, p=0.4)
    A = rand(n, n) .< p
    A = A .* (1 .- I(n))  # no self-loops
    L = Diagonal(sum(A, dims=2)[:]) - A
    return sparse(L)
end



function interlayer_adjacency_balanced(N1, N2, k_links_per_node=1)
    W = spzeros(N1 + N2, N1 + N2)
    for i in 1:N1
        js = rand(1:N2, k_links_per_node)
        for j in js
            j_idx = N1 + j
            W[i, j_idx] = 1.0
            W[j_idx, i] = 1.0
        end
    end
    return W
end



function laplacian_from_adjacency(W)
    d = vec(sum(W, dims=2))
    return Diagonal(d) - W
end



function multilayer_laplacian()
    L1 = ring_laplacian(N1)
    L2 = random_laplacian(N2)
    W_inter = interlayer_adjacency_balanced(N1, N2)
    L_inter = laplacian_from_adjacency(W_inter)

    L = spzeros(N1+N2, N1+N2)
    L[1:N1,1:N1] = L1
    L[N1+1:end, N1+1:end] = L2
    L += D * L_inter
    return L
end


function consensus_dynamics!(du, u, p, t)
    global last_switch_time = p[:last_switch_time]
    global current_L = p[:current_L]

    if t - last_switch_time >= Δt_switch
        current_L .= multilayer_laplacian()
        p[:last_switch_time] = t
    end

    n = N1+N2
    du[1:n] = u[n+1:end]
	du[n+1:end] = -( current_L +  I )*u[n+1:end] - current_L^2*u[1:n]
end


function solve_ode()
    N = N1+N2
    u0 = vcat( 1.0:1.0:N, zeros(N) )
    current_L = multilayer_laplacian()
    p = Dict(:current_L => current_L, :last_switch_time => 0.0)
    prob = ODEProblem( consensus_dynamics!, u0, (0.0, T_max), p )
    sol = solve( prob, DP5() )
end


function make_array( sol )
    data = Array{Float64}( undef, length(sol.u), length(sol.u[1]) )

    for (i,ci) in enumerate(sol.u)
        data[i, :] =  ci
    end

    return data
end


function plot_solution()
    c = pyx.canvas.canvas()


    #---------------
    sol1 = solve_ode()
    data1 = make_array(sol1)

    g1 = pyx.graph.graphxy(
        width = 6, height = 2.5,
        x = pyx.graph.axis.lin( min = 0, max = 20, title = "Time (au)" ),
        y = pyx.graph.axis.lin(min = 0, max=9, title = raw"$y_k$"),
    )


    for (i, si) in enumerate( eachcol( data1[ :, 1:(N1+N2) ] ) )
        if i <= N1
            color = pyx.color.cmyk.Blue
        else
            color = pyx.color.cmyk.Red
        end

        g1.plot(
            pyx.graph.data.values( x = sol1.t, y = si ),
            [
                pyx.graph.style.line([ color, pyx.style.linewidth.thick ])
            ]
        )
    end

    for xi in 0:2.5:20
        g1.layers["background"].stroke( pyx.path.line(g1.pos(xi,0)..., g1.pos(xi,9)...),
                  [ pyx.style.linestyle.dashed, pyx.color.gray(0.5) ]
        )
    end


    #---------------
    sol2 = solve_ode()
    data2 = make_array(sol2)

    g2 = pyx.graph.graphxy(
        width = 6, height = 2.5,
        x = pyx.graph.axis.linkedaxis( g1.axes["x"] ),
        y = pyx.graph.axis.lin(min = 0, max=9, title = raw"$y_k$"),
    )


    for (i, si) in enumerate( eachcol( data2[ :, 1:(N1+N2) ] ) )
        if i <= N1
            color = pyx.color.cmyk.Blue
        else
            color = pyx.color.cmyk.Red
        end

        g2.plot(
            pyx.graph.data.values( x = sol2.t, y = si ),
            [
                pyx.graph.style.line([ color, pyx.style.linewidth.thick ])
            ]
        )
    end

    for xi in 0:2.5:20
        g2.layers["background"].stroke( pyx.path.line(g2.pos(xi,0)..., g2.pos(xi,9)...),
                  [ pyx.style.linestyle.dashed, pyx.color.gray(0.5) ]
        )
    end


    #---------------
    sol3 = solve_ode()
    data3 = make_array(sol3)

    g3 = pyx.graph.graphxy(
        width = 6, height = 2.5,
        x = pyx.graph.axis.linkedaxis( g1.axes["x"] ),
        y = pyx.graph.axis.lin(min = 0, max=9, title = raw"$y_k$"),
    )


    for (i, si) in enumerate( eachcol( data3[ :, 1:(N1+N2) ] ) )
        if i <= N1
            color = pyx.color.cmyk.Blue
        else
            color = pyx.color.cmyk.Red
        end

        g3.plot(
            pyx.graph.data.values( x = sol3.t, y = si ),
            [
                pyx.graph.style.line([ color, pyx.style.linewidth.thick ])
            ]
        )
    end

    for xi in 0:2.5:20
        g3.layers["background"].stroke( pyx.path.line(g3.pos(xi,0)..., g3.pos(xi,9)...),
                  [ pyx.style.linestyle.dashed, pyx.color.gray(0.5) ]
        )
    end


    g1.text( g1.pos(18,7.5)..., "(c)" )
    g2.text( g2.pos(18,7.5)..., "(b)" )
    g3.text( g3.pos(18,7.5)..., "(a)" )

    c.insert( g1)
    c.insert( g2, [pyx.trafo.translate(0,3)])
    c.insert( g3, [pyx.trafo.translate(0,6)])
    c.writePDFfile("switching_multilayer.pdf")
end

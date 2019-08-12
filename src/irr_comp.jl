using HomotopyContinuation

"""
    `irred_comp(witn::WSet)`

Input is a witness set representing a pure dimensional algebraic set.
Computes a partition of the given witness set into subsets, that represent
the irreducible components of the algebraic set.
"""
function irred_comp(witn::WSet; K::Integer=8)
    #compute traces:
    traces = trace_list(witn)

    if abs(sum(trace_list)) > 0.00001
        error("witn does not form a complete witness set!")
    end

    Graph = []
    for i = 1:length(witn.coords)
        make_set(i, Graph, traces[i], coords[i])
    end

    for index = 1:K
        perm = perm_sol(witn)
        for i = 1:length(perm)
            for j = 1:length(Graph)
                if sum(abs.(perm[i] - witn.coords[j])) < 0.000001
                    Union(Graph[i], Graph[j])
                end
            end
        end
        filter!(x == x.parent, Graph)
    end

    components = [get_component(node) for node in Graph if node.parent == node]
    traces = [  sum([trace_list[i]  for i in comp])  for comp in components  ]
    return(components, traces)
end

"""
    `perm_sol(witn)`

`witn::WSet` represents a witness set.
Its solutions are tracked along a random change of coordinates
of the underlying affine linear space.
"""
function perm_sol(witn::WSet)
    @polyvar λ[1:size(witn.matrix, 2)]
    @polyvar s
    matrix1, matrix2 = randn(ComplexF32, size(witn.matrix), 2)
    transl1, transl2 = randn(ComplexF32, size(witn.matrix, 1), 2)

    G = witn.matrix * λ + witn.transl
    G_1 = matrix1 * λ + transl1
    G_2 = matrix2 * λ + transl2

    new_coords = solutions(solve(F ∘ ((1-s)*G   + s*G_1), witn.coords; parameters = [s], start_parameters = [0], target_parameters = [1] ))
    new_coords = solutions(solve(F ∘ ((1-s)*G_1 + s*G_2), new_coords;  parameters = [s], start_parameters = [0], target_parameters = [1] ))
    new_coords = solutions(solve(F ∘ ((1-s)*G_2 + s*G  ), new_coords;  parameters = [s], start_parameters = [0], target_parameters = [1] ))
    return new_coords
    #Alles in eine Homotopie?!
    #Was, wenn wir Lösungen verlieren?!
end

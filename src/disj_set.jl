
function make_set(x, Graph, trace, coord)
    new_node = Node(x, nothing, 0, 1, [], trace, coord)
    new_node.parent = new_node
    push!(Graph, new_node)
end

mutable struct Node
    index
    parent
    rank
    size
    childs
    trace
    coord
end

function find!(node::Node)
    while node.parent != node
        if node.parent.parent != node.parent
            push!(node.parent.parent.childs, node)
            node.parent.childs = filter!(x->x != node, node.parent.childs)
            node.parent = node.parent.parent
        end
        node = node.parent
    end
    return(node)
end

function union!(x, y)
    xRoot = find!(x)
    yRoot = find!(y)
    if xRoot == yRoot
        return
    end
    if xRoot.rank < yRoot.rank
        xRoot, yRoot = yRoot, xRoot # swap xRoot and yRoot
    end
# merge yRoot into xRoot
    yRoot.parent = xRoot
    push!(xRoot.childs, yRoot)
    if xRoot.rank == yRoot.rank
        xRoot.rank = xRoot.rank + 1
    end
end

function get_component!(x::Node)
    x = find!(x)
    return recurs(x)
end
function recurs(x::Node)
    list = [x.index]
    for child in x.childs
        list = vcat(list, recurs(child))
    end
    return list
end


function get_trace!(x::Node)
    x = find!(x)
    return trace!(x, 0)
end

function trace!(x::Node, result)
    result += x.trace
    for child in x.childs
        result = trace!(child, result)
    end
    return result
end

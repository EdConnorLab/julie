import igraph as ig

# Read the agonism data for April and August
agonism = ig.Graph.Read_Ncol("ZombiesAgonismSept.csv", directed=True)

# Calculate degrees
deg = agonism.degree(mode="all")

# Plot the agonism network
ig.plot(agonism, vertex_size=[d * 1 for d in deg], vertex_color=[(0.1, 0.7, 0.8, 0.5)], edge_arrow_size=0.4, layout=agonism.layout_kamada_kawai())

# Simplify the agonism network
agonism_simple = agonism.simplify()

# Plot the simplified agonism network
ig.plot(agonism_simple, vertex_size=[d * 1.5 for d in deg], vertex_label_size=0.8, vertex_color="lightyellow", edge_arrow_size=0.4, layout=agonism_simple.layout_kamada_kawai())

# Read the new agonistic graph
am = ig.Graph.Read_Adjacency("ZombiesAgonismMatrix.csv", mode="directed", attribute="weight")

# Plot the new agonistic graph
ig.plot(am, edge_width=[e["weight"] for e in am.es], edge_curved=[-0.2 + i * 0.2 / am.ecount() for i in range(am.ecount())], layout=am.layout_kamada_kawai())

# Read the new affiliation graph
af = ig.Graph.Read_Adjacency("ZombiesAffiliationMatrix.csv", mode="undirected", attribute="weight")

# Plot the new affiliation graph
ig.plot(af, edge_width=[e["weight"] for e in af.es], edge_curved=[-0.2 + i * 0.2 / af.ecount() for i in range(af.ecount())])

# Read the affiliation data for April and August
affiliation = ig.Graph.Read_Ncol("ZombiesAffiliationSNA.csv", directed=False)

# Calculate degrees
deg = affiliation.degree(mode="all")

# Plot the affiliation network
ig.plot(affiliation, vertex_size=[d * 1 for d in deg], vertex_color=[(0.1, 0.7, 0.8, 0.5)], edge_arrow_size=0.4, layout=affiliation.layout_kamada_kawai())

# Simplify the affiliation network
affiliation_simple = affiliation.simplify()

# Plot the simplified affiliation network
ig.plot(affiliation_simple, vertex_size=[d * 0.6 for d in deg], vertex_label_size=0.8, vertex_color="lightpink", edge_arrow_size=0.4, layout=affiliation_simple.layout_kamada_kawai())
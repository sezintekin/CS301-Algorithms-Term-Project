#include <iostream>
#include <vector>
#include <set>
#include <utility>
#include <ctime>
#include <cstdlib>
#include <chrono>
using namespace std;

// this struct named Edge is to represent the edges which has two vertices u, v and a its weight w for our graph

struct Edge {
    int u, v, w;
};



//-------------------------------------------------------------------------------------------------------------------------------------

class Graph {
public:
    int n;
    vector<Edge> edges;

    Graph(int n) : n(n) {} // constructor for our graph

    // function to add the edges to our graph
    void add_edge(int u, int v, int w) {
        edges.push_back({u, v, w});
    }
};

//-------------------------------------------------------------------------------------------------------------------------------------

class DisjointSet {
public:
    
    vector<int> parent, rank;
    DisjointSet(int n) {
        parent.resize(n);
        rank.resize(n);
        for (int i = 0; i < n; i++) {
            parent[i] = i;
            rank[i] = 0;
        }
    }

    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    void union_sets(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        if (rootX != rootY) {
            if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else {
                parent[rootY] = rootX;
                rank[rootX] += 1;
            }
        }
    }
};


//-------------------------------------------------------------------------------------------------------------------------------------

Graph generate_random_graph(int n, int m) {
    
    Graph G(n); // creates a graph with n vertices

    set<pair<int, int>> edge_set; // a set of edges to keep track of the edges that have been added to our graph

    while (edge_set.size() < m) {
        int u = rand() % n; // initializes a vertice which has a random value between 0 and n-1
        int v = rand() % n;

        if (u != v && edge_set.find({u, v}) == edge_set.end() && edge_set.find({v, u}) == edge_set.end()) { // checks if u and v are not equal and if the edge {u, v} is not in the graph
            int w = rand() % 100 + 1; // generates a random weight for the edge between 1 and 100
            G.add_edge(u, v, w); // adds the vertices and its edge to the graph
            edge_set.insert({u, v}); // adds the edge set {u, v} ro the edge set
        }
    }

    return G;
}

//------------------------------------------------------------------------------------------

vector<vector<int>> adjacency_list(const Graph& G) {
    
    vector<vector<int>> adj(G.n); // creates a vector of vectors and initializes it with G.n empty vectors

    for (const auto& e : G.edges) { // iterates over each Edge object in the edges vector
        adj[e.u].push_back(e.v); // adds the endpoints of the current edge e to each other's list of neighbors
        adj[e.v].push_back(e.u);
    }

    return adj;
}

//------------------------------------------------------------------------------------------

bool check_degree_constraint(const vector<vector<int>>& adj, int k) {
    
    for (const auto& neighbors : adj) { // iterates over each vector of integers (neighbors) in the input vector of vectors adj
        if (neighbors.size() > k) { // checks if the degree of the current vector is greater than constraint k
            return false;
        }
    }
    return true;
}

//------------------------------------------------------------------------------------------

void dfs_spanning_trees(const Graph& G, vector<vector<int>>& adj, vector<bool>& visited, int u, vector<Graph>& spanning_trees) {
    visited[u] = true;
    bool is_spanning_tree = true;

    for (int v : adj[u]) { // iterates over each neighbor of v of the current vertex u in adj list
        if (!visited[v]) { // if not visited
            is_spanning_tree = false;
            vector<vector<int>> new_adj = adj; // creates a new list new_adj by adding an edge between u and v, calls dfs_spanning_trees recursively with the updated adjacency list
            new_adj[u].push_back(v);
            new_adj[v].push_back(u);
            dfs_spanning_trees(G, new_adj, visited, v, spanning_trees);
        }
    }

    if (is_spanning_tree) { // if the current tree is  a spanning tree
        Graph tree(G.n); // creates a new object as tree which has G.n vertices
        for (int i = 0; i < G.n; ++i) {
            for (int j : adj[i]) {
                if (i < j) {
                    tree.add_edge(i, j, 0); // adds edges to it based on the adjacency list adj
                }
            }
        }

        spanning_trees.push_back(tree); // appends it to the vector of Graph objects named spanning_trees
    }

    visited[u] = false; // this is because if the current vertex is still marked as visited, the function may not explore all possible trees

}

//------------------------------------------------------------------------------------------

vector<Graph> generate_all_spanning_trees(const Graph& G) {
    
    vector<vector<int>> adj = adjacency_list(G); // calls the adjacency_list function defined earlier to generate the adjacency list to create a vector of vectors

    vector<bool> visited(G.n, false); // this is for to keep track of visited vertices during the generation of spanning trees
    vector<Graph> spanning_trees; // this vector will be used to store the generated spanning trees

    dfs_spanning_trees(G, adj, visited, 0, spanning_trees); // calls the function to generate all possible spanning trees

    return spanning_trees; // returns the vector of Graph objects spanning_trees, which contains all possible spanning trees of the input graph G

}

//------------------------------------------------------------------------------------------

int brute_force_algorithm(const Graph& G) {
    
    for (int k = 1; k < G.n; k++) {
        vector<Graph> all_spanning_trees = generate_all_spanning_trees(G); // this line creates a vector of Graph objects and initializes it with all possible spanning trees

        for (const auto& tree : all_spanning_trees) { // iterates over each Graph object tree in the vector of Graph objects all_spanning_trees
            vector<vector<int>> adj = adjacency_list(tree); // creates a vector of vectors of integers named adj and initializes it with the adjacency list of the current spanning tree

            if (check_degree_constraint(adj, k)) { // if the degree constraint is satisfied it returns the k value
                return k;
            }
        }
    }

    return -1;
}

//-----------------------------------------------------------------------------------------

bool can_add_edge(const vector<vector<int>>& adj, int node, int k) { // checks if an edge can be added to a node in a grap
    return adj[node].size() < k;
}

//------------------------------------------------------------------------------------------

bool heuristicalgorithm(const Graph& G, int k) {
    
    vector<Edge> sortedEdges = G.edges; // creates a vector and initializes it with the edges from the input graph G
    
    sort(sortedEdges.begin(), sortedEdges.end(), [](const Edge& a, const Edge& b) {
        return a.w < b.w;
    }); // sorts the edges based on their weights in ascending order using

    Graph MST(G.n);  // a new Graph object is created with the same number of nodes as G
    DisjointSet ds(G.n); // DisjointSet object is also created with the same number of nodes as G to keep track of disjoint sets
    vector<vector<int>> adj(MST.n);

    for (const Edge& edge : sortedEdges) {
        if (ds.find(edge.u) != ds.find(edge.v) && can_add_edge(adj, edge.u, k) && can_add_edge(adj, edge.v, k)) { // it checks if the two endpoints belong to different sets in the disjoint set data structure (
            ds.union_sets(edge.u, edge.v);
            MST.add_edge(edge.u, edge.v, edge.w); // the edge is added to the minimum spanning tree
            adj[edge.u].push_back(edge.v);
            adj[edge.v].push_back(edge.u);
        }
    }

    // if the constructed graph is a spanning tree of G
    int root = ds.find(0);
    for (int i = 1; i < G.n; i++) {
        if (ds.find(i) != root) {
            return false; // not all nodes are connected, so there's no spanning tree
        }
    }

    return true; // a degree-constrained spanning tree exists for the given k
}


//------------------------------------------------------------------------------------------

int find_smallest_k(Graph& G) {
    int left = 1, right = G.n;
    while (left < right) {
        int mid = left + (right - left) / 2; // calculates the midpoint between left and right using binary search.
        if (heuristicalgorithm(G, mid)) { // calls the heuristicalgorithm function with the current value of mid as the degree constraint k
            right = mid; // a degree-constrained spanning tree exists for the given mid
        } else {
            left = mid + 1; // a degree-constrained spanning tree does not exist for the given mid
        }
    }
    return left; // represents the smallest value of k found during the search
}

//------------------------------------------------------------------------------------------


int main() {
    
    srand(time(nullptr)); // this is for generating different random numbers each time the code runs

//    for (int i = 0; i < 15; i++) {
//        int n = rand() % 16 + 5;
//        int max_edges = n * (n - 1) / 2;
//        int m = rand() % (max_edges + 1);
//
//        Graph G = generate_random_graph(n, m);
//
//        cout << "Generated graph " << i + 1 << " with " << n << " vertices and " << m << " edges" << endl;
//
//        int k = brute_force_algorithm(G);
//
//        if (k == -1) {
//            cout << "No suitable solution found" << endl;
//        } else {
//            cout << "The smallest integer k such that G has a spanning tree in which no node has degree greater than k is: " << k << endl;
//        }
//
//        cout << endl;
//    }

//    cout << "Generating a solution using the heuristic algorithm: " << endl;
//
//    for (int i = 0; i < 15; i++) {
//        int n = rand() % 16 + 5;
//        int max_edges = n * (n - 1) / 2;
//        int m = rand() % (max_edges + 1);
//
//        Graph G = generate_random_graph(n, m);
//
//        cout << "Generated graph " << i + 1 << " with " << n << " vertices and " << m << " edges" << endl;
//
//        auto start = std::chrono::high_resolution_clock::now();
//        int k = find_smallest_k(G);
//        auto end = std::chrono::high_resolution_clock::now();
//
//        if (k > G.n) {
//            cout << "No suitable solution found" << endl;
//        } else {
//            cout << "The smallest integer k such that G has a spanning tree in which no node has degree greater than k is: " << k << endl;
//        }
//
//        std::chrono::duration<double> elapsed = end - start;
//        cout << "The algorithm took " << elapsed.count() << " seconds to run." << endl;
//        cout << endl;
//    }

    
    // Number of iterations and maximum graph size
    int num_iterations = 1;
    int max_graph_size = 10;

    cout << "Graph Size vs. Runtime Analysis" << endl << endl;

    // Brute-force algorithm
    cout << "Brute-Force Algorithm:" << endl;

    for (int graph_num = 1; graph_num <= 5; graph_num++) {
        cout << "Graph " << graph_num << ":" << endl;

        for (int n = 5; n <= max_graph_size; n++) {
            int max_edges = n * (n - 1) / 2;
            double total_runtime = 0.0;

            for (int i = 0; i < num_iterations; i++) {
                int m = rand() % (max_edges + 1);
                Graph G = generate_random_graph(n, m);

                auto start = chrono::high_resolution_clock::now();
                brute_force_algorithm(G);
                auto end = chrono::high_resolution_clock::now();

                chrono::duration<double> elapsed = end - start;
                total_runtime += elapsed.count();
            }

            double average_runtime = total_runtime / num_iterations;
            cout << "Graph Size: " << n << " - Average Runtime: " << average_runtime << " seconds" << endl;
        }

        cout << endl;
    }

    cout << endl;

    // Heuristic algorithm
    cout << "Heuristic Algorithm:" << endl;

    for (int graph_num = 1; graph_num <= 5; graph_num++) {
        cout << "Graph " << graph_num << ":" << endl;

        for (int n = 5; n <= max_graph_size; n++) {
            int max_edges = n * (n - 1) / 2;
            double total_runtime = 0.0;

            for (int i = 0; i < num_iterations; i++) {
                int m = rand() % (max_edges + 1);
                Graph G = generate_random_graph(n, m);

                auto start = chrono::high_resolution_clock::now();
                find_smallest_k(G);
                auto end = chrono::high_resolution_clock::now();

                chrono::duration<double> elapsed = end - start;
                total_runtime += elapsed.count();
            }

            double average_runtime = total_runtime / num_iterations;
            cout << "Graph Size: " << n << " - Average Runtime: " << average_runtime << " seconds" << endl;
        }

        cout << endl;
    }

    
    return 0;
}

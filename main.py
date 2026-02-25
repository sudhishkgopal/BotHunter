import networkx as nx
import matplotlib.pyplot as plt
import multiprocessing as mp
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os

# FASTAPI APP SETUP

app = FastAPI(title="BotHunter API", description="Bot Detection using K-Core Algorithm")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# FUNCTION DEFINITIONS

def get_k_core(graph, k):
    """LOGIC: The K-Core Pruning Function"""
    # Convert to a dictionary adjacency list for speed O(1) lookup time
    adj = {node: set(neighbors) for node, neighbors in graph.adjacency()}
    
    while True:
        # Find nodes that have fewer than K neighbors
        to_remove = [node for node, neighbors in adj.items() if len(neighbors) < k]
        
        if not to_remove:
            break 
            
        for node in to_remove:
            # Remove this node from its neighbors' adjacency lists
            for neighbor in adj[node]:
                adj[neighbor].remove(node)
            # Remove the node itself
            del adj[node]
            
    return nx.Graph(adj)  # Convert back to NetworkX for visualization


def find_nodes_to_remove(node_chunk, adj_dictionary, k):
    """Looks at a chunk of nodes at a time and finds which ones have <k connections"""
    return [node for node in node_chunk if len(adj_dictionary[node]) < k]


def get_k_core_parallel(graph, k):
    """Parallel K-Core implementation for large graphs"""
    # Convert to a dictionary adjacency list for speed O(1) lookup time
    adj = {node: set(neighbors) for node, neighbors in graph.adjacency()}
    nodes = list(adj.keys())
    num_cores = mp.cpu_count()  # Gets number of CPU cores available
    
    while True:
        # Split the nodes into chunks so each core processes a chunk, resulting in faster processing
        chunk_size = len(nodes) // num_cores  # Find chunk size
        # Start at index i and go up to i + chunk_size
        chunks = [nodes[i:i + chunk_size] for i in range(0, len(nodes), chunk_size)]
        # Chunk 1: nodes[0,chunk_size], Chunk 2: nodes[chunk_size+1, 2*chunk_size], etc.

        # Now parallel the processing
        with mp.Pool(processes=num_cores) as pool:
            # Each core processes its designated chunk
            results = pool.starmap(find_nodes_to_remove, [(chunk, adj, k) for chunk in chunks])
        
        # Project the list of lists into a single list of suspected bots
        nodes_to_remove = [node for sublist in results for node in sublist]

        # if there are no suspected bots return early, otherwise continue
        if not nodes_to_remove:
            break

        for node in nodes_to_remove:  # Loop through each suspected bot
            if node in adj:  # Check if node exists, it could have already been removed by another neighbor
                for neighbor in adj[node]:  # Tell neighbors to remove node from their adj list
                    if node in adj[neighbor]:   
                        adj[neighbor].remove(node)
                del adj[node]  # Once neighbors have deleted the node, delete the node itself

        nodes = list(adj.keys())  # Update remaining nodes
        print("Nodes remaining:", len(nodes))
        
    return nx.Graph(adj)  # Convert back to NetworkX for visualization


def save_visualization(original_G, core_G, filename="bot_detection.png"):
    """Save visualization to file instead of showing"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), facecolor='#F5F5F5')
    
    # Use 'Spring Layout' to position nodes further apart to see the "structure"
    pos_orig = nx.spring_layout(original_G, k=0.15, iterations=50, seed=42)
    pos_core = nx.shell_layout(core_G) if len(core_G.nodes()) > 0 else {}
    
    # Draw Original Network
    nx.draw_networkx_nodes(original_G, pos_orig, node_size=25, 
                           node_color='#3498db', alpha=0.8, ax=ax1)
    nx.draw_networkx_edges(original_G, pos_orig, width=0.5, 
                           edge_color='grey', alpha=0.2, ax=ax1)  # Low alpha = cleaner
    ax1.set_title("Original Network: The Noise", fontsize=20, fontweight='bold')
    ax1.axis('off')

    # Draw Bot Clique
    if len(core_G.nodes()) > 0:
        nx.draw_networkx_nodes(core_G, pos_core, node_size=150, 
                               node_color='#e74c3c', edgecolors='black', ax=ax2)
        nx.draw_networkx_edges(core_G, pos_core, width=1.5, 
                               edge_color='#c0392b', alpha=0.6, ax=ax2)
    ax2.set_title("Detected Bot Core: The Signal", fontsize=20, fontweight='bold', color='#c0392b')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def load_twitter_data(file_path):
    """Load Twitter network data from file"""
    print("Loading Twitter data from:", file_path)  # Loading Standford SNAP Dataset information by Jure Leskovec
    print("Opening file")
    # Using nx.read_edgelist to optimize space
    # Twitter file uses spaces between IDs
    twitter_graph = nx.read_edgelist(file_path, create_using=nx.Graph(), nodetype=int)
    return twitter_graph


# API ENDPOINTS

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to BotHunter API",
        "version": "1.0",
        "endpoints": {
            "/simulate": "POST - Run bot detection on simulated network",
            "/analyze": "POST - Analyze uploaded network file",
            "/twitter": "GET - Analyze Twitter dataset",
            "/download/{filename}": "GET - Download generated files"
        }
    }


@app.post("/simulate")
async def simulate_bot_detection(num_humans: int = 100, num_bots: int = 15, k: int = 10):
    """
    Run bot detection on a simulated network
    
    Parameters:
    - num_humans: Number of human nodes (default: 100)
    - num_bots: Number of bot nodes (default: 15)
    - k: K-core threshold (default: 10)
    """
    try:
        # Generating crowd of humans and bots
        # Create random humans and bots
        human_network = nx.erdos_renyi_graph(num_humans, 0.05)  # Connected humans graph (each human has a 5% chance of connecting to another)
        bot_network = nx.complete_graph(num_bots)  # Bot connections (all bots are connected)
        # Assigning nodes to each ID, shifting over so bot IDs don't overlap with human IDs
        bot_network = nx.relabel_nodes(bot_network, {i: i + num_humans for i in range(num_bots)})
        human_network = nx.compose(human_network, bot_network)  # Merging humans and bots together
        
        # RUN bot detection
        bot_core = get_k_core(human_network, k=k)
        
        # Save visualization
        save_visualization(human_network, bot_core, "simulation_result.png")
        
        # Get results
        bot_list = list(bot_core.nodes())
        
        # Save to JSON
        with open("simulation_bots.json", "w") as f:
            json.dump(bot_list, f)
        
        return JSONResponse({
            "status": "success",
            "total_nodes": human_network.number_of_nodes(),
            "total_edges": human_network.number_of_edges(),
            "detected_bots": len(bot_list),
            "bot_ids": bot_list,
            "visualization": "/download/simulation_result.png",
            "results_file": "/download/simulation_bots.json"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze_network(file: UploadFile = File(...), k: int = 10, use_parallel: bool = False):
    """
    Analyze an uploaded network file
    
    Parameters:
    - file: Network edge list file (txt format)
    - k: K-core threshold (default: 10)
    - use_parallel: Use parallel processing for large networks (default: False)
    """
    try:
        # Save uploaded file temporarily
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Load network
        network = nx.read_edgelist(temp_file, create_using=nx.Graph(), nodetype=int)
        
        print(f"Loaded network with {network.number_of_nodes()} nodes and {network.number_of_edges()} edges")
        
        # Run bot detection
        if use_parallel:
            bot_core = get_k_core_parallel(network, k=k)
        else:
            bot_core = get_k_core(network, k=k)
        
        # Save visualization (skip for very large networks)
        if network.number_of_nodes() < 5000:
            save_visualization(network, bot_core, "analysis_result.png")
            viz_path = "/download/analysis_result.png"
        else:
            viz_path = "Skipped (network too large for visualization)"
        
        # Get results
        bot_list = list(bot_core.nodes())
        
        # Save to JSON
        with open("detected_bots.json", "w") as f:
            json.dump(bot_list, f)
        
        # Clean up temp file
        os.remove(temp_file)
        
        return JSONResponse({
            "status": "success",
            "total_nodes": network.number_of_nodes(),
            "total_edges": network.number_of_edges(),
            "detected_bots": len(bot_list),
            "bot_ids": bot_list[:100],  # Return first 100 for display
            "visualization": viz_path,
            "results_file": "/download/detected_bots.json"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/twitter")
async def analyze_twitter(k: int = 10, use_parallel: bool = True):
    """
    Analyze Twitter dataset
    
    Parameters:
    - k: K-core threshold (default: 10)
    - use_parallel: Use parallel processing (default: True)
    """
    try:
        file_path = "twitter_combined.txt"  # File path for functional Twitter dataset
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404, 
                detail="Twitter dataset not found. Please add twitter_combined.txt to the project directory."
            )
        
        # Load Twitter data
        twitter_network = load_twitter_data(file_path)
        
        print(f"Successfully loaded {twitter_network.number_of_nodes()} users")
        print(f"Successfully loaded {twitter_network.number_of_edges()} connections")
        
        # Run bot detection
        if use_parallel:
            bot_core = get_k_core_parallel(twitter_network, k=k)
        else:
            bot_core = get_k_core(twitter_network, k=k)
        
        # Get results
        bot_list = list(bot_core.nodes())
        
        # Save to JSON
        with open("twitter_bots.json", "w") as f:
            json.dump(bot_list, f)
        
        print(f"Saved {len(bot_list)} bot IDs to twitter_bots.json")
        
        # Note: Skip visualization for very large networks to save time
        if twitter_network.number_of_nodes() < 10000:
            save_visualization(twitter_network, bot_core, "twitter_result.png")
            viz_path = "/download/twitter_result.png"
        else:
            viz_path = "Skipped (network too large for visualization)"
        
        return JSONResponse({
            "status": "success",
            "total_nodes": twitter_network.number_of_nodes(),
            "total_edges": twitter_network.number_of_edges(),
            "detected_bots": len(bot_list),
            "bot_ids": bot_list[:100],  # Return first 100 for display
            "visualization": viz_path,
            "results_file": "/download/twitter_bots.json"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download generated files (images, JSON results)
    """
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(filename, filename=filename)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "BotHunter API"}



# MAIN EXECUTION (for testing without API)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
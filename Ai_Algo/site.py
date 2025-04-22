import streamlit as st
import numpy as np
import time
import heapq
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import random
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle

# Create a multi-page app
st.set_page_config(layout="wide", page_title="Search Algorithms Visualization", page_icon="🔍")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Visualization", "Comparison", "Resources"])

if page == "Visualization":
    st.title("Search Algorithms Visualization")
    st.markdown("""
    This application visualizes common search algorithms (BFS, DFS, UCS, A*) on a grid.
    Watch how different algorithms explore the space step by step to find a path from start to goal!
    """)

    # Global variables
    GRID_SIZE = 15
    WALL_PROBABILITY = 0.2

    # Create session state if not exists
    if 'grid' not in st.session_state:
        st.session_state.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        st.session_state.start = (0, 0)
        st.session_state.goal = (GRID_SIZE-1, GRID_SIZE-1)
        st.session_state.current_challenge = None
        st.session_state.solution_found = False
        st.session_state.exploration_path = []
        st.session_state.final_path = []
        st.session_state.explored_nodes_count = 0
        st.session_state.current_step = 0
        st.session_state.animation_running = False
        st.session_state.animation_speed = 0.5  # seconds per step
        # For visualization of the search tree
        st.session_state.search_tree = nx.DiGraph()
        st.session_state.frontier = []
        st.session_state.visited_nodes = set()

    # Function to reset grid
    def reset_grid():
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        # Add random walls
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if random.random() < WALL_PROBABILITY and (i, j) != st.session_state.start and (i, j) != st.session_state.goal:
                    grid[i, j] = 1  # 1 represents a wall
        
        st.session_state.grid = grid
        st.session_state.exploration_path = []
        st.session_state.final_path = []
        st.session_state.explored_nodes_count = 0
        st.session_state.solution_found = False
        st.session_state.current_step = 0
        st.session_state.animation_running = False
        st.session_state.search_tree = nx.DiGraph()
        st.session_state.frontier = []
        st.session_state.visited_nodes = set()

    # Function to create a predefined challenge
    def create_challenge(challenge_type):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        
        if challenge_type == "maze":
            # Create a maze-like structure
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    if i % 2 == 0 and j % 2 == 0:
                        grid[i, j] = 1
            
            # Create paths through the maze
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    if i % 4 == 0 or j % 4 == 0:
                        grid[i, j] = 0
            
        elif challenge_type == "spiral":
            # Create a spiral pattern
            center = GRID_SIZE // 2
            for i in range(1, center):
                # Top horizontal wall
                grid[center-i, center-i:center+i+1] = 1
                # Right vertical wall
                grid[center-i:center+i+1, center+i] = 1
                # Bottom horizontal wall
                grid[center+i, center-i:center+i+1] = 1
                # Left vertical wall
                grid[center-i:center+i+1, center-i] = 1
                
                # Create openings
                grid[center-i, center-i+1] = 0  # Top wall opening
                grid[center+i-1, center+i] = 0  # Right wall opening
                grid[center+i, center+i-1] = 0  # Bottom wall opening
                grid[center-i+1, center-i] = 0  # Left wall opening
        
        elif challenge_type == "islands":
            # Create several disconnected "islands" with narrow passages
            for i in range(1, GRID_SIZE-1, 4):
                for j in range(1, GRID_SIZE-1, 4):
                    # Create a block island
                    block_size = min(3, GRID_SIZE-i-1, GRID_SIZE-j-1)
                    grid[i:i+block_size, j:j+block_size] = 1
                    
                    # Create a random opening
                    opening = random.randint(0, 3)
                    if opening == 0 and i > 0:
                        grid[i-1, j+block_size//2] = 0  # Top opening
                    elif opening == 1 and j+block_size < GRID_SIZE:
                        grid[i+block_size//2, j+block_size] = 0  # Right opening
                    elif opening == 2 and i+block_size < GRID_SIZE:
                        grid[i+block_size, j+block_size//2] = 0  # Bottom opening
                    elif opening == 3 and j > 0:
                        grid[i+block_size//2, j-1] = 0  # Left opening
        
        # Ensure start and goal are not walls
        grid[st.session_state.start] = 0
        grid[st.session_state.goal] = 0
        
        st.session_state.grid = grid
        st.session_state.current_challenge = challenge_type
        st.session_state.exploration_path = []
        st.session_state.final_path = []
        st.session_state.explored_nodes_count = 0
        st.session_state.solution_found = False
        st.session_state.current_step = 0
        st.session_state.animation_running = False
        st.session_state.search_tree = nx.DiGraph()
        st.session_state.frontier = []
        st.session_state.visited_nodes = set()

    # Helper function to check if a position is valid
    def is_valid(grid, pos):
        i, j = pos
        return 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1] and grid[i, j] != 1

    # Manhattan distance heuristic for A*
    def manhattan_distance(pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    # Function to update search tree visualization
    def update_search_tree(parent, child):
        if parent is not None and child is not None:
            st.session_state.search_tree.add_edge(parent, child)

    # BFS algorithm with search tree visualization
    def bfs(grid, start, goal):
        queue = deque([start])
        visited = {start: None}
        exploration_path = []
        frontier_history = []  # Track frontier at each step
        
        # Initialize search tree with start node
        st.session_state.search_tree.add_node(start)
        
        while queue:
            current = queue.popleft()
            exploration_path.append(current)
            frontier_history.append(list(queue))  # Save current frontier state
            
            if current == goal:
                st.session_state.solution_found = True
                break
            
            # Explore neighbors
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_pos = (current[0] + di, current[1] + dj)
                if is_valid(grid, next_pos) and next_pos not in visited:
                    queue.append(next_pos)
                    visited[next_pos] = current
                    # Update search tree
                    st.session_state.search_tree.add_node(next_pos)
                    st.session_state.search_tree.add_edge(current, next_pos)
        
        # Add final frontier state
        frontier_history.append(list(queue))
        
        # Reconstruct path
        path = []
        if st.session_state.solution_found:
            current = goal
            while current:
                path.append(current)
                current = visited[current]
            path.reverse()
        
        return exploration_path, path, len(exploration_path), frontier_history

    # DFS algorithm with search tree visualization
    def dfs(grid, start, goal):
        stack = [start]
        visited = {start: None}
        exploration_path = []
        frontier_history = []  # Track frontier at each step
        
        # Initialize search tree with start node
        st.session_state.search_tree.add_node(start)
        
        while stack:
            current = stack.pop()
            exploration_path.append(current)
            frontier_history.append(list(stack))  # Save current frontier state
            
            if current == goal:
                st.session_state.solution_found = True
                break
            
            # Explore neighbors in reverse order (for better visualization)
            for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                next_pos = (current[0] + di, current[1] + dj)
                if is_valid(grid, next_pos) and next_pos not in visited:
                    stack.append(next_pos)
                    visited[next_pos] = current
                    # Update search tree
                    st.session_state.search_tree.add_node(next_pos)
                    st.session_state.search_tree.add_edge(current, next_pos)
        
        # Add final frontier state
        frontier_history.append(list(stack))
        
        # Reconstruct path
        path = []
        if st.session_state.solution_found:
            current = goal
            while current:
                path.append(current)
                current = visited[current]
            path.reverse()
        
        return exploration_path, path, len(exploration_path), frontier_history

    # UCS algorithm with search tree visualization
    def ucs(grid, start, goal):
        pq = [(0, start)]  # (cost, position)
        visited = {start: (None, 0)}  # position: (parent, cost)
        exploration_path = []
        frontier_history = []  # Track frontier at each step
        
        # Initialize search tree with start node
        st.session_state.search_tree.add_node(start)
        
        while pq:
            cost, current = heapq.heappop(pq)
            
            if current in [node for _, node in frontier_history[-1]] if frontier_history else []:
                continue
                
            exploration_path.append(current)
            frontier_history.append([(c, pos) for c, pos in pq])  # Save current frontier state
            
            if current == goal:
                st.session_state.solution_found = True
                break
            
            # Explore neighbors
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_pos = (current[0] + di, current[1] + dj)
                # Each step costs 1
                new_cost = cost + 1
                
                if is_valid(grid, next_pos) and (next_pos not in visited or new_cost < visited[next_pos][1]):
                    heapq.heappush(pq, (new_cost, next_pos))
                    visited[next_pos] = (current, new_cost)
                    # Update search tree
                    st.session_state.search_tree.add_node(next_pos)
                    st.session_state.search_tree.add_edge(current, next_pos)
        
        # Add final frontier state
        frontier_history.append([(c, pos) for c, pos in pq])
        
        # Reconstruct path
        path = []
        if st.session_state.solution_found:
            current = goal
            while current:
                path.append(current)
                current = visited[current][0]
            path.reverse()
        
        return exploration_path, path, len(exploration_path), frontier_history

    # A* algorithm with search tree visualization
    def astar(grid, start, goal):
        pq = [(0, 0, start)]  # (f_score, g_score, position)
        g_score = {start: 0}  # g_score[n] is the cost from start to n
        f_score = {start: manhattan_distance(start, goal)}  # f_score[n] = g_score[n] + h(n)
        visited = {start: None}  # position: parent
        exploration_path = []
        frontier_history = []  # Track frontier at each step
        
        # Initialize search tree with start node
        st.session_state.search_tree.add_node(start)
        
        while pq:
            _, cost, current = heapq.heappop(pq)
            
            if current in [node for _, _, node in frontier_history[-1]] if frontier_history else []:
                continue
                
            exploration_path.append(current)
            frontier_history.append([(f, g, pos) for f, g, pos in pq])  # Save current frontier state
            
            if current == goal:
                st.session_state.solution_found = True
                break
            
            # Explore neighbors
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_pos = (current[0] + di, current[1] + dj)
                
                if is_valid(grid, next_pos):
                    tentative_g = g_score[current] + 1
                    
                    if next_pos not in g_score or tentative_g < g_score[next_pos]:
                        visited[next_pos] = current
                        g_score[next_pos] = tentative_g
                        f_score[next_pos] = tentative_g + manhattan_distance(next_pos, goal)
                        heapq.heappush(pq, (f_score[next_pos], tentative_g, next_pos))
                        # Update search tree
                        st.session_state.search_tree.add_node(next_pos)
                        st.session_state.search_tree.add_edge(current, next_pos)
        
        # Add final frontier state
        frontier_history.append([(f, g, pos) for f, g, pos in pq])
        
        # Reconstruct path
        path = []
        if st.session_state.solution_found:
            current = goal
            while current:
                path.append(current)
                current = visited[current]
            path.reverse()
        
        return exploration_path, path, len(exploration_path), frontier_history

    # Function to display the grid with current progress
    def display_grid(grid, start, goal, exploration_path=None, final_path=None, current_step=0, frontier=None):
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create a colormap
        cmap = mcolors.ListedColormap(['white', 'black', 'lightskyblue', 'green', 'red', 'orange', 'yellow'])
        bounds = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Start with a copy of the grid
        display_grid = grid.copy()
    
    # Mark frontier nodes
        if frontier and current_step < len(frontier):
           frontier_nodes = frontier[current_step]
        # Check if frontier_nodes is not empty before accessing its elements
           if frontier_nodes and len(frontier_nodes) > 0:
            # Check the type of the first element to determine the algorithm
            first_node = frontier_nodes[0]
            
            if isinstance(first_node, tuple):
                if len(first_node) == 2:  # BFS, DFS
                    for pos in frontier_nodes:
                        if pos != start and pos != goal:
                            display_grid[pos] = 6  # Yellow for frontier
                elif len(first_node) == 2 and isinstance(first_node[0], (int, float)):  # UCS
                    for _, pos in frontier_nodes:
                        if pos != start and pos != goal:
                            display_grid[pos] = 6  # Yellow for frontier
                elif len(first_node) == 3:  # A*
                    for _, _, pos in frontier_nodes:
                        if pos != start and pos != goal:
                            display_grid[pos] = 6  # Yellow for frontier
    
    # Mark exploration path up to current step
        if exploration_path:
            for i in range(min(current_step + 1, len(exploration_path))):
              pos = exploration_path[i]
              if pos != start and pos != goal:
                  display_grid[pos] = 2  # Blue for explored
    
    # Mark final path if solution is found and we're at the end of exploration
        if final_path and current_step >= len(exploration_path) - 1:
          for pos in final_path:
              if pos != start and pos != goal:
                  display_grid[pos] = 3  # Green for path
    
    # Mark start and goal
        display_grid[start] = 4  # Red for start
        display_grid[goal] = 5  # Orange for goal
    
    # Display grid
        ax.imshow(display_grid, cmap=cmap, norm=norm)
    
    # Add grid lines
        for i in range(grid.shape[0] + 1):
           ax.axhline(i - 0.5, color='gray', linestyle='-', linewidth=1)
           ax.axvline(i - 0.5, color='gray', linestyle='-', linewidth=1)
    
    # Highlight current node being explored
        if exploration_path and current_step < len(exploration_path):
           current_node = exploration_path[current_step]
           rect = Rectangle((current_node[1]-0.5, current_node[0]-0.5), 1, 1, linewidth=3, edgecolor='magenta', facecolor='none')
           ax.add_patch(rect)
    
    # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add legend
        legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='white', markersize=15, label='Empty Space'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=15, label='Wall'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightskyblue', markersize=15, label='Explored'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow', markersize=15, label='Frontier'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=15, label='Path'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=15, label='Start'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=15, label='Goal'),]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=8)
    
    # Add step counter
        if exploration_path:
           step_text = f"Step: {min(current_step + 1, len(exploration_path))}/{len(exploration_path)}"
           ax.text(0.5, -0.15, step_text, transform=ax.transAxes, ha='center', fontsize=12)
    
        return fig

    # Function to display the search tree
    def display_search_tree(search_tree, current_node=None, path=None):
        if not search_tree.nodes():
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title("Search Tree (Empty)")
            ax.axis('off')
            return fig
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Position nodes using a hierarchical layout
        pos = nx.spring_layout(search_tree)
        
        # Draw nodes
        nx.draw_networkx_nodes(search_tree, pos, node_size=300, node_color='lightblue', ax=ax)
        
        # Highlight the current node
        if current_node and current_node in search_tree.nodes:
            nx.draw_networkx_nodes(search_tree, pos, nodelist=[current_node], node_size=300, node_color='red', ax=ax)
        
        # Highlight the path
        if path:
            # Create path edges
            path_edges = []
            for i in range(len(path)-1):
                if search_tree.has_edge(path[i], path[i+1]) or search_tree.has_edge(path[i+1], path[i]):
                    path_edges.append((path[i], path[i+1]))
            
            nx.draw_networkx_edges(search_tree, pos, edgelist=path_edges, width=2, edge_color='green', ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(search_tree, pos, width=1, edge_color='gray', ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(search_tree, pos, font_size=8, ax=ax)
        
        ax.set_title("Search Tree")
        ax.axis('off')
        
        return fig

    # Function to display the frontier data structure
    def display_frontier(frontier, algorithm):
        if not frontier or not frontier[-1]:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "Frontier is empty", ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Get current frontier
        current_frontier = frontier[-1]
        
        # Format frontier based on algorithm
        if algorithm == "BFS":
            title = "Queue (FIFO)"
            frontier_str = "Front " + " → ".join([f"({x}, {y})" for x, y in current_frontier]) + " Back"
        elif algorithm == "DFS":
            title = "Stack (LIFO)"
            frontier_str = "Top " + " → ".join([f"({x}, {y})" for x, y in reversed(current_frontier)]) + " Bottom"
        elif algorithm == "UCS":
            title = "Priority Queue (by cost)"
            frontier_str = " | ".join([f"({x}, {y}) cost={c}" for c, (x, y) in sorted(current_frontier)])
        elif algorithm == "A*":
            title = "Priority Queue (by f=g+h)"
            frontier_str = " | ".join([f"({x}, {y}) f={f}, g={g}" for f, g, (x, y) in sorted(current_frontier)])
        else:
            title = "Frontier"
            frontier_str = str(current_frontier)
        
        ax.text(0.5, 0.7, title, ha='center', fontsize=14, fontweight='bold')
        ax.text(0.5, 0.3, frontier_str, ha='center', fontsize=10, wrap=True)
        ax.axis('off')
        
        return fig

    # Main layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Controls")
        
        # Algorithm selection
        algorithm = st.selectbox(
            "Select algorithm",
            ["BFS", "DFS", "UCS", "A*"]
        )
        
        # Challenge selection
        challenge_type = st.selectbox(
            "Select challenge",
            ["Random", "Maze", "Spiral", "Islands"]
        )
        
        # Button to create a new challenge
        if st.button("Generate New Challenge"):
            if challenge_type.lower() == "random":
                reset_grid()
            else:
                create_challenge(challenge_type.lower())
        
        # Button to run algorithm
        if st.button("Run Algorithm"):
            st.session_state.exploration_path = []
            st.session_state.final_path = []
            st.session_state.solution_found = False
            st.session_state.current_step = 0
            st.session_state.search_tree = nx.DiGraph()
            st.session_state.frontier = []
            
            if algorithm == "BFS":
                st.session_state.exploration_path, st.session_state.final_path, st.session_state.explored_nodes_count, st.session_state.frontier = bfs(
                    st.session_state.grid, st.session_state.start, st.session_state.goal
                )
            elif algorithm == "DFS":
                st.session_state.exploration_path, st.session_state.final_path, st.session_state.explored_nodes_count, st.session_state.frontier = dfs(
                    st.session_state.grid, st.session_state.start, st.session_state.goal
                )
            elif algorithm == "UCS":
                st.session_state.exploration_path, st.session_state.final_path, st.session_state.explored_nodes_count, st.session_state.frontier = ucs(
                    st.session_state.grid, st.session_state.start, st.session_state.goal
                )
            elif algorithm == "A*":
                st.session_state.exploration_path, st.session_state.final_path, st.session_state.explored_nodes_count, st.session_state.frontier = astar(
                    st.session_state.grid, st.session_state.start, st.session_state.goal
                )
        
        # Animation controls
        st.subheader("Animation Controls")
        
        # Animation speed
        animation_speed = st.slider(
            "Animation Speed (seconds per step)",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1
        )
        st.session_state.animation_speed = animation_speed
        
        # Step navigation
        col_prev, col_play, col_next = st.columns(3)
        
        with col_prev:
            if st.button("Previous Step"):
                if st.session_state.current_step > 0:
                    st.session_state.current_step -= 1
                    st.session_state.animation_running = False
        
        with col_play:
            if not st.session_state.animation_running:
                if st.button("Play Animation"):
                    st.session_state.animation_running = True
            else:
                if st.button("Pause Animation"):
                    st.session_state.animation_running = False
        
        with col_next:
            if st.button("Next Step"):
                if st.session_state.exploration_path and st.session_state.current_step < len(st.session_state.exploration_path) - 1:
                    st.session_state.current_step += 1
                    st.session_state.animation_running = False
        
        # Reset to beginning
        if st.button("Reset Animation"):
            st.session_state.current_step = 0
            st.session_state.animation_running = False
        
        # Display statistics
        st.subheader("Statistics")
        
        if st.session_state.exploration_path:
            st.write(f"Nodes explored so far: {min(st.session_state.current_step + 1, len(st.session_state.exploration_path))}/{len(st.session_state.exploration_path)}")
            
            if st.session_state.frontier and st.session_state.current_step < len(st.session_state.frontier):
                frontier_size = len(st.session_state.frontier[st.session_state.current_step])
                st.write(f"Frontier size: {frontier_size}")
            
            if st.session_state.solution_found:
                st.write(f"Final path length: {len(st.session_state.final_path)}")
                
                if st.session_state.current_step >= len(st.session_state.exploration_path) - 1:
                    st.success("Solution found!")
            elif st.session_state.current_step >= len(st.session_state.exploration_path) - 1:
                st.error("No solution exists!")
        
        # Algorithm explanation
        st.subheader("Algorithm Explanation")
        
        if algorithm == "BFS":
            st.markdown("""
            **Breadth-First Search (BFS)**
            
            BFS explores all neighbors at the present depth prior to moving on to nodes at the next depth level.
            
            **Key characteristics:**
            - Data structure: **Queue** (FIFO)
            - Complete: Always finds a solution if one exists
            - Optimal: Finds shortest path when edges have equal cost
            - Time complexity: O(b^d)
            - Space complexity: O(b^d)
            
            BFS expands like ripples in a pond, exploring everything at the same distance before moving farther away.
            """)
        
        elif algorithm == "DFS":
            st.markdown("""
            **Depth-First Search (DFS)**
            
            DFS explores as far as possible along each branch before backtracking.
            
            **Key characteristics:**
            - Data structure: **Stack** (LIFO)
            - Complete: Only if search space is finite
            - Not optimal: Doesn't guarantee shortest path
            - Time complexity: O(b^m)
            - Space complexity: O(bm)
            
            DFS dives deep into one path, only backtracking when it hits a dead end or fully explores a branch.
            """)
        
        elif algorithm == "UCS":
            st.markdown("""
            **Uniform Cost Search (UCS)**
            
            UCS expands the node with the lowest path cost, using a priority queue.
            
            **Key characteristics:**
            - Data structure: **Priority Queue** (by path cost)
            - Complete: Yes, if costs are positive
            - Optimal: Yes, finds least-cost path
            - Time complexity: O(b^(C*/ε)) where C* is optimal path cost
            - Space complexity: O(b^(C*/ε))
            
            UCS always chooses the lowest-cost unexpanded node, guaranteeing optimal paths.
            """)
        
        elif algorithm == "A*":
            st.markdown("""
            **A* Search**
            
            A* combines UCS with a heuristic function to direct the search toward the goal.
            
            **Key characteristics:**
            - Data structure: **Priority Queue** (by f = g + h)
            - Complete: Yes, if heuristic is admissible
            - Optimal: Yes, if heuristic is admissible
            - Time complexity: O(b^d) in worst case
            - Space complexity: O(b^d) in worst case
            
            A* is informed search using both path cost and estimated distance to goal, making it very efficient.
            """)

    with col2:
        st.header("Visualization")
        
        # Grid display placeholder
        grid_placeholder = st.empty()
        
        # Display grid
        if st.session_state.exploration_path:
            current_node = st.session_state.exploration_path[st.session_state.current_step] if st.session_state.current_step < len(st.session_state.exploration_path) else None
            fig = display_grid(
                st.session_state.grid, 
                st.session_state.start, 
                st.session_state.goal,
                st.session_state.exploration_path,
                st.session_state.final_path,
                st.session_state.current_step,
                st.session_state.frontier
            )
            grid_placeholder.pyplot(fig)
            plt.close(fig)
        else:
            fig = display_grid(st.session_state.grid, st.session_state.start, st.session_state.goal)
            grid_placeholder.pyplot(fig)
            plt.close(fig)
        
        # Display search tree and frontier side by side
        col_tree, col_frontier = st.columns(2)
        
        with col_tree:
            st.subheader("Search Tree")
            
            if st.session_state.exploration_path:
                current_node = st.session_state.exploration_path[st.session_state.current_step] if st.session_state.current_step < len(st.session_state.exploration_path) else None
                tree_fig = display_search_tree(
                    st.session_state.search_tree,
                    current_node,
                    st.session_state.final_path if st.session_state.solution_found and st.session_state.current_step >= len(st.session_state.exploration_path) - 1 else None
                )
                st.pyplot(tree_fig)
                plt.close(tree_fig)
            else:
                tree_fig = display_search_tree(st.session_state.search_tree)
                st.pyplot(tree_fig)
                plt.close(tree_fig)
        
        with col_frontier:
            st.subheader("Frontier Data Structure")
            
            if st.session_state.frontier:
                frontier_fig = display_frontier(st.session_state.frontier[:st.session_state.current_step+1], algorithm)
                st.pyplot(frontier_fig)
                plt.close(frontier_fig)
            else:
                frontier_fig = display_frontier([], algorithm)
                st.pyplot(frontier_fig)
                plt.close(frontier_fig)
    
    # Animation logic
    if st.session_state.animation_running and st.session_state.exploration_path:
        if st.session_state.current_step < len(st.session_state.exploration_path) - 1:
            time.sleep(st.session_state.animation_speed)
            st.session_state.current_step += 1
            st.rerun()
        else:
            st.session_state.animation_running = False
            st.rerun()

elif page == "Comparison":
    st.title("Algorithm Comparison")
    st.markdown("""
    Compare the performance of different search algorithms on the same grid.
    """)
    
    # Grid size
    grid_size = st.slider("Grid Size", 5, 20, 15)
    
    # Wall probability
    wall_prob = st.slider("Wall Probability", 0.0, 0.5, 0.2)
    
    # Create grid
    if 'comparison_grid' not in st.session_state or st.button("Generate New Grid"):
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Add random walls
        for i in range(grid_size):
            for j in range(grid_size):
                if random.random() < wall_prob and (i, j) != (0, 0) and (i, j) != (grid_size-1, grid_size-1):
                    grid[i, j] = 1  # 1 represents a wall
        
        st.session_state.comparison_grid = grid
        st.session_state.comparison_start = (0, 0)
        st.session_state.comparison_goal = (grid_size-1, grid_size-1)
        st.session_state.comparison_results = {}
    
    # Helper functions duplicated for comparison
    def is_valid_comp(grid, pos):
        i, j = pos
        return 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1] and grid[i, j] != 1

    def manhattan_distance_comp(pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    # Algorithm implementations
    def bfs_comp(grid, start, goal):
        queue = deque([start])
        visited = {start: None}
        exploration_path = []
        expanded_nodes = 0
        
        while queue:
            current = queue.popleft()
            exploration_path.append(current)
            expanded_nodes += 1
            
            if current == goal:
                break
            
            # Explore neighbors
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_pos = (current[0] + di, current[1] + dj)
                if is_valid_comp(grid, next_pos) and next_pos not in visited:
                    queue.append(next_pos)
                    visited[next_pos] = current
        
        # Reconstruct path
        path = []
        if goal in visited:
            current = goal
            while current:
                path.append(current)
                current = visited[current]
            path.reverse()
        
        return path, expanded_nodes, len(exploration_path)
    
    def dfs_comp(grid, start, goal):
        stack = [start]
        visited = {start: None}
        exploration_path = []
        expanded_nodes = 0
        
        while stack:
            current = stack.pop()
            exploration_path.append(current)
            expanded_nodes += 1
            
            if current == goal:
                break
            
            # Explore neighbors
            for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                next_pos = (current[0] + di, current[1] + dj)
                if is_valid_comp(grid, next_pos) and next_pos not in visited:
                    stack.append(next_pos)
                    visited[next_pos] = current
        
        # Reconstruct path
        path = []
        if goal in visited:
            current = goal
            while current:
                path.append(current)
                current = visited[current]
            path.reverse()
        
        return path, expanded_nodes, len(exploration_path)
    
    def ucs_comp(grid, start, goal):
        pq = [(0, start)]
        visited = {start: (None, 0)}
        exploration_path = []
        expanded_nodes = 0
        visited_set = set()
        
        while pq:
            cost, current = heapq.heappop(pq)
            
            if current in visited_set:
                continue
            
            visited_set.add(current)
            exploration_path.append(current)
            expanded_nodes += 1
            
            if current == goal:
                break
            
            # Explore neighbors
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_pos = (current[0] + di, current[1] + dj)
                # Each step costs 1
                new_cost = cost + 1
                
                if is_valid_comp(grid, next_pos) and (next_pos not in visited or new_cost < visited[next_pos][1]):
                    heapq.heappush(pq, (new_cost, next_pos))
                    visited[next_pos] = (current, new_cost)
        
        # Reconstruct path
        path = []
        if goal in visited:
            current = goal
            while current:
                path.append(current)
                current = visited[current][0]
            path.reverse()
        
        return path, expanded_nodes, len(exploration_path)
    
    def astar_comp(grid, start, goal):
        pq = [(0, 0, start)]
        g_score = {start: 0}
        f_score = {start: manhattan_distance_comp(start, goal)}
        visited = {start: None}
        exploration_path = []
        expanded_nodes = 0
        visited_set = set()
        
        while pq:
            _, cost, current = heapq.heappop(pq)
            
            if current in visited_set:
                continue
            
            visited_set.add(current)
            exploration_path.append(current)
            expanded_nodes += 1
            
            if current == goal:
                break
            
            # Explore neighbors
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_pos = (current[0] + di, current[1] + dj)
                
                if is_valid_comp(grid, next_pos):
                    tentative_g = g_score[current] + 1
                    
                    if next_pos not in g_score or tentative_g < g_score[next_pos]:
                        visited[next_pos] = current
                        g_score[next_pos] = tentative_g
                        f_score[next_pos] = tentative_g + manhattan_distance_comp(next_pos, goal)
                        heapq.heappush(pq, (f_score[next_pos], tentative_g, next_pos))
        
        # Reconstruct path
        path = []
        if goal in visited:
            current = goal
            while current:
                path.append(current)
                current = visited[current]
            path.reverse()
        
        return path, expanded_nodes, len(exploration_path)
    
    # Run comparison
    if st.button("Run Comparison"):
        results = {}
        
        # Run BFS
        path, expanded, explored = bfs_comp(st.session_state.comparison_grid, st.session_state.comparison_start, st.session_state.comparison_goal)
        results["BFS"] = {
            "path": path,
            "path_length": len(path) if path else 0,
            "expanded_nodes": expanded,
            "explored_nodes": explored,
            "solution_found": bool(path)
        }
        
        # Run DFS
        path, expanded, explored = dfs_comp(st.session_state.comparison_grid, st.session_state.comparison_start, st.session_state.comparison_goal)
        results["DFS"] = {
            "path": path,
            "path_length": len(path) if path else 0,
            "expanded_nodes": expanded,
            "explored_nodes": explored,
            "solution_found": bool(path)
        }
        
        # Run UCS
        path, expanded, explored = ucs_comp(st.session_state.comparison_grid, st.session_state.comparison_start, st.session_state.comparison_goal)
        results["UCS"] = {
            "path": path,
            "path_length": len(path) if path else 0,
            "expanded_nodes": expanded,
            "explored_nodes": explored,
            "solution_found": bool(path)
        }
        
        # Run A*
        path, expanded, explored = astar_comp(st.session_state.comparison_grid, st.session_state.comparison_start, st.session_state.comparison_goal)
        results["A*"] = {
            "path": path,
            "path_length": len(path) if path else 0,
            "expanded_nodes": expanded,
            "explored_nodes": explored,
            "solution_found": bool(path)
        }
        
        st.session_state.comparison_results = results
    
    # Display the grid
    if 'comparison_grid' in st.session_state:
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create a colormap
        cmap = mcolors.ListedColormap(['white', 'black', 'red', 'orange'])
        bounds = [0, 0.5, 1.5, 2.5, 3.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # Start with a copy of the grid
        display_grid = st.session_state.comparison_grid.copy()
        
        # Mark start and goal
        display_grid[st.session_state.comparison_start] = 2  # Red for start
        display_grid[st.session_state.comparison_goal] = 3  # Orange for goal
        
        # Display grid
        ax.imshow(display_grid, cmap=cmap, norm=norm)
        
        # Add grid lines
        for i in range(st.session_state.comparison_grid.shape[0] + 1):
            ax.axhline(i - 0.5, color='gray', linestyle='-', linewidth=1)
            ax.axvline(i - 0.5, color='gray', linestyle='-', linewidth=1)
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add legend  
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='white', markersize=15, label='Empty Space'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=15, label='Wall'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=15, label='Start'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=15, label='Goal'),
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
        
        st.pyplot(fig)
        plt.close(fig)
    
    # Display comparison results
    if 'comparison_results' in st.session_state and st.session_state.comparison_results:
        st.subheader("Comparison Results")
        
        # Create a table
        data = []
        for algo, result in st.session_state.comparison_results.items():
            data.append({
                "Algorithm": algo,
                "Solution Found": "Yes" if result["solution_found"] else "No",
                "Path Length": result["path_length"] if result["solution_found"] else "N/A",
                "Nodes Expanded": result["expanded_nodes"],
                "Nodes Explored": result["explored_nodes"]
            })
        
        # Display as a dataframe
        st.table(data)
        
        # Create comparison charts
        st.subheader("Visual Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Path Length Comparison
            path_lengths = [result["path_length"] if result["solution_found"] else 0 for algo, result in st.session_state.comparison_results.items()]
            algorithms = list(st.session_state.comparison_results.keys())
            
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(algorithms, path_lengths)
            
            # Color bars based on optimality (minimum path length)
            if any(path_lengths):
                min_path = min([l for l in path_lengths if l > 0])
                for i, length in enumerate(path_lengths):
                    if length == min_path:
                        bars[i].set_color('green')
                    elif length == 0:
                        bars[i].set_color('red')
                    else:
                        bars[i].set_color('orange')
            
            ax.set_title("Path Length Comparison")
            ax.set_ylabel("Path Length")
            ax.set_ylim(bottom=0)
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom')
                else:
                    ax.text(bar.get_x() + bar.get_width()/2., 0.5,
                            'N/A', ha='center', va='bottom')
            
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            # Nodes Explored Comparison
            explored_nodes = [result["explored_nodes"] for algo, result in st.session_state.comparison_results.items()]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(algorithms, explored_nodes)
            
            # Color bars based on efficiency (minimum nodes explored)
            if any(explored_nodes):
                min_explored = min(explored_nodes)
                for i, explored in enumerate(explored_nodes):
                    if explored == min_explored:
                        bars[i].set_color('green')
                    elif explored > min_explored * 2:
                        bars[i].set_color('red')
                    else:
                        bars[i].set_color('orange')
            
            ax.set_title("Nodes Explored Comparison")
            ax.set_ylabel("Number of Nodes")
            ax.set_ylim(bottom=0)
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
            
            st.pyplot(fig)
            plt.close(fig)
        
        # Visualize paths on separate grids
        st.subheader("Path Visualization")
        
        cols = st.columns(len(st.session_state.comparison_results))
        
        for i, (algo, result) in enumerate(st.session_state.comparison_results.items()):
            with cols[i]:
                st.write(f"**{algo}**")
                
                fig, ax = plt.subplots(figsize=(6, 6))
                
                # Create a colormap
                cmap = mcolors.ListedColormap(['white', 'black', 'lightskyblue', 'green', 'red', 'orange'])
                bounds = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
                norm = mcolors.BoundaryNorm(bounds, cmap.N)
                
                # Start with a copy of the grid
                display_grid = st.session_state.comparison_grid.copy()
                
                # Mark explored nodes
                for node in result.get("path", []):
                    if node != st.session_state.comparison_start and node != st.session_state.comparison_goal:
                        display_grid[node] = 3  # Green for path
                
                # Mark start and goal
                display_grid[st.session_state.comparison_start] = 4  # Red for start
                display_grid[st.session_state.comparison_goal] = 5  # Orange for goal
                
                # Display grid
                ax.imshow(display_grid, cmap=cmap, norm=norm)
                
                # Add grid lines
                for j in range(st.session_state.comparison_grid.shape[0] + 1):
                    ax.axhline(j - 0.5, color='gray', linestyle='-', linewidth=1)
                    ax.axvline(j - 0.5, color='gray', linestyle='-', linewidth=1)
                
                # Remove ticks
                ax.set_xticks([])
                ax.set_yticks([])
                
                st.pyplot(fig)
                plt.close(fig)
                
                if result["solution_found"]:
                    st.write(f"Path length: {result['path_length']}")
                else:
                    st.write("No solution found")
        
        # Analysis
        st.subheader("Analysis")
        
        # Determine which algorithm found the shortest path
        solution_paths = {algo: result["path_length"] for algo, result in st.session_state.comparison_results.items() if result["solution_found"]}
        
        if solution_paths:
            min_path_length = min(solution_paths.values())
            optimal_algorithms = [algo for algo, length in solution_paths.items() if length == min_path_length]
            
            st.write(f"**Optimal solution found by:** {', '.join(optimal_algorithms)} (path length: {min_path_length})")
        else:
            st.write("**No solution was found by any algorithm.**")
        
        # Determine which algorithm was most efficient
        explored_nodes = {algo: result["explored_nodes"] for algo, result in st.session_state.comparison_results.items()}
        min_explored = min(explored_nodes.values())
        most_efficient_algorithms = [algo for algo, count in explored_nodes.items() if count == min_explored]
        
        st.write(f"**Most efficient algorithm:** {', '.join(most_efficient_algorithms)} (explored {min_explored} nodes)")
        
        # Summary based on expected algorithm behaviors
        st.markdown("""
        **Expected algorithm behaviors:**
        
        1. **BFS:** Guarantees shortest path when all edge costs are equal. Explores nodes level by level.
        
        2. **DFS:** May not find shortest path. Often explores fewer nodes than BFS if multiple solutions exist, but can get stuck in deep paths.
        
        3. **UCS:** Guarantees optimal path even with varying edge costs (though all edges have cost=1 in this grid).
        
        4. **A*:** Usually most efficient when a good heuristic is available. Manhattan distance is a perfect heuristic for grid movement without diagonal moves.
        """)

elif page == "Resources":
    st.title("Learning Resources")
    
    st.header("Search Algorithms Overview")
    st.markdown("""
    Search algorithms are fundamental to artificial intelligence and computer science. They allow us to find paths from an initial state to a goal state in a search space.
    
    ### Key Concepts
    
    - **Search Space**: The set of all possible states that can be reached from the initial state.
    - **Search Tree**: A tree representation of the search process, where each node represents a state and each edge represents an action.
    - **Frontier**: The set of nodes that have been discovered but not yet explored.
    - **Explored Set**: The set of nodes that have been fully explored.
    - **Path Cost**: The cost of reaching a node from the start node.
    - **Heuristic Function**: A function that estimates the cost to reach the goal from a given node.
    
    ### Common Search Algorithms
    
    1. **Uninformed Search Algorithms**:
       - Breadth-First Search (BFS)
       - Depth-First Search (DFS)
       - Uniform Cost Search (UCS)
       
    2. **Informed Search Algorithms**:
       - Greedy Best-First Search
       - A* Search
       - Iterative Deepening A* (IDA*)
    """)
    
    st.header("Online Resources")
    st.markdown("""
    ### Interactive Learning
    
    - [VisuAlgo](https://visualgo.net/) - Visualizing algorithms and data structures
    - [PathFinding.js](https://qiao.github.io/PathFinding.js/visual/) - A comprehensive path-finding library with visualization
    
    ### Books and Courses
    
    - "Artificial Intelligence: A Modern Approach" by Stuart Russell and Peter Norvig
    - [Stanford CS221: Artificial Intelligence](https://stanford-cs221.github.io/)
    - [edX: Artificial Intelligence](https://www.edx.org/learn/artificial-intelligence)
    
    ### Video Tutorials
    
    - [MIT OpenCourseWare: Search Algorithms](https://www.youtube.com/watch?v=gGQ-vAmdAOI)
    - [Computerphile: A* Search Algorithm](https://www.youtube.com/watch?v=ySN5Wnu88nE)
    """)
    
    st.header("Key Differences Between Algorithms")
    
    comparison_data = {
        "Algorithm": ["BFS", "DFS", "UCS", "A*"],
        "Data Structure": ["Queue (FIFO)", "Stack (LIFO)", "Priority Queue (by cost)", "Priority Queue (by f = g + h)"],
        "Complete?": ["Yes", "Only if search space is finite", "Yes (if costs > 0)", "Yes (if heuristic is admissible)"],
        "Optimal?": ["Yes (for uniform costs)", "No", "Yes", "Yes (if heuristic is admissible)"],
        "Space Complexity": ["O(b^d)", "O(bm)", "O(b^(C*/ε))", "O(b^d)"],
        "Time Complexity": ["O(b^d)", "O(b^m)", "O(b^(C*/ε))", "O(b^d)"]
    }
    
    st.table(comparison_data)
    
    st.header("Heuristics for Grid-Based Search")
    st.markdown("""
    In grid-based pathfinding, several heuristics are commonly used:
    
    1. **Manhattan Distance**: |x1 - x2| + |y1 - y2|
       - Appropriate for grids with no diagonal movement
    
    2. **Euclidean Distance**: sqrt((x1 - x2)² + (y1 - y2)²)
       - Appropriate for spaces where any direction of movement is allowed
    
    3. **Chebyshev Distance**: max(|x1 - x2|, |y1 - y2|)
       - Appropriate for grids with diagonal movement where diagonal move cost = straight move cost
    
    4. **Octile Distance**: max(|x1 - x2|, |y1 - y2|) + (sqrt(2) - 1) * min(|x1 - x2|, |y1 - y2|)
       - Appropriate for grids with diagonal movement where diagonal move cost = sqrt(2) * straight move cost
    """)
    
    st.header("Applications")
    st.markdown("""
    Search algorithms have numerous applications:
    
    - **Pathfinding in video games**
    - **Robot navigation**
    - **Web crawlers**
    - **Puzzle solving**
    - **Natural language processing**
    - **Automated planning**
    
    The choice of algorithm depends on the specific requirements of the problem, such as optimality, time constraints, and memory limitations.
    """)
    
    st.header("About This App")
    st.markdown("""
    This application was created to help students and enthusiasts learn about search algorithms through interactive visualization. By watching how different algorithms explore the grid, you can develop an intuition for their behavior and understand their strengths and weaknesses.
    
    ### Features
    
    - **Step-by-step visualization** of algorithm execution
    - **Search tree visualization** to understand how algorithms build and explore the search space
    - **Frontier visualization** to see how different data structures affect algorithm behavior
    - **Multiple algorithms** including BFS, DFS, UCS, and A*
    - **Various challenge grids** to test algorithms in different scenarios
    - **Performance comparison** to analyze efficiency and optimality
    
    ### How to Use
    
    1. Select an algorithm and challenge type
    2. Generate a new challenge grid
    3. Run the algorithm
    4. Use the animation controls to step through the execution
    5. Compare different algorithms on the same grid in the Comparison tab
    
    ### Implementation Details
    
    The app is built using Python and Streamlit, with visualization powered by Matplotlib and NetworkX. The search algorithms are implemented from scratch to clearly demonstrate their behavior.
    """)

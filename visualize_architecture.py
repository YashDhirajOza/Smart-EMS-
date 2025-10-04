"""
System Architecture Visualization
Generates a diagram showing all components and their interactions
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Microgrid EMS RL System Architecture', 
        ha='center', va='top', fontsize=18, fontweight='bold')

# ============================================
# Layer 1: Data Sources (Bottom)
# ============================================
data_y = 1.0
data_items = [
    ('Solar Plant\nData', 1, '#FFD700'),
    ('Weather\nData', 2.5, '#87CEEB'),
    ('Load\nProfiles', 4, '#90EE90'),
    ('Price\nData', 5.5, '#FFB6C1'),
    ('EV Fleet\nPatterns', 7, '#DDA0DD')
]

for label, x, color in data_items:
    rect = FancyBboxPatch((x-0.4, data_y-0.3), 0.8, 0.6, 
                          boxstyle="round,pad=0.05", 
                          facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x, data_y, label, ha='center', va='center', fontsize=9, fontweight='bold')

# ============================================
# Layer 2: Data Processing
# ============================================
proc_y = 2.5
proc_rect = FancyBboxPatch((0.5, proc_y-0.4), 8, 0.8,
                           boxstyle="round,pad=0.1",
                           facecolor='#E0E0E0', edgecolor='black', linewidth=2)
ax.add_patch(proc_rect)
ax.text(4.5, proc_y, 'Data Preprocessing Module', ha='center', va='center', 
        fontsize=11, fontweight='bold')
ax.text(4.5, proc_y-0.25, '(Aggregation, Scaling, Normalization)', 
        ha='center', va='center', fontsize=8, style='italic')

# Arrows from data to processing
for label, x, color in data_items:
    arrow = FancyArrowPatch((x, data_y+0.3), (x, proc_y-0.45),
                           arrowstyle='->', mutation_scale=15, 
                           linewidth=1.5, color='gray')
    ax.add_artist(arrow)

# ============================================
# Layer 3: Core Components
# ============================================
core_y = 4.5
core_components = [
    ('Battery\nDegradation\nModel', 1.2, '#FF6B6B'),
    ('EV Fleet\nSimulator', 2.8, '#4ECDC4'),
    ('Microgrid\nEnvironment\n(Gym)', 5, '#45B7D1'),
    ('Safety\nSupervisor', 7.2, '#FFA07A'),
    ('Reward\nCalculator', 8.8, '#98D8C8')
]

for label, x, color in core_components:
    rect = FancyBboxPatch((x-0.6, core_y-0.5), 1.2, 1.0,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x, core_y, label, ha='center', va='center', 
            fontsize=9, fontweight='bold')

# Arrow from processing to environment
arrow = FancyArrowPatch((4.5, proc_y+0.4), (5, core_y-0.55),
                       arrowstyle='->', mutation_scale=20, 
                       linewidth=2, color='black')
ax.add_artist(arrow)

# ============================================
# Layer 4: RL Agent
# ============================================
agent_y = 6.5
agent_rect = FancyBboxPatch((3.5, agent_y-0.5), 3, 1.0,
                           boxstyle="round,pad=0.1",
                           facecolor='#6C5CE7', edgecolor='black', linewidth=3)
ax.add_patch(agent_rect)
ax.text(5, agent_y+0.15, 'RL Agent (PPO)', ha='center', va='center', 
        fontsize=12, fontweight='bold', color='white')
ax.text(5, agent_y-0.15, 'Actor-Critic Network', ha='center', va='center', 
        fontsize=9, style='italic', color='white')

# Arrows between agent and environment
# Observation: Environment -> Agent
arrow_obs = FancyArrowPatch((5, core_y+0.55), (5, agent_y-0.55),
                           arrowstyle='->', mutation_scale=25, 
                           linewidth=3, color='#2ECC71')
ax.add_artist(arrow_obs)
ax.text(5.5, (core_y+agent_y)/2, 'Observation\n(~100 dims)', 
        ha='left', va='center', fontsize=9, color='#2ECC71', fontweight='bold')

# Action: Agent -> Environment
arrow_act = FancyArrowPatch((4.5, agent_y-0.55), (4.5, core_y+0.55),
                           arrowstyle='->', mutation_scale=25, 
                           linewidth=3, color='#E74C3C')
ax.add_artist(arrow_act)
ax.text(3.9, (core_y+agent_y)/2, 'Action\n(5 dims)', 
        ha='right', va='center', fontsize=9, color='#E74C3C', fontweight='bold')

# Reward: Environment -> Agent
arrow_rew = FancyArrowPatch((5.5, core_y+0.55), (5.5, agent_y-0.55),
                           arrowstyle='->', mutation_scale=25, 
                           linewidth=3, color='#F39C12')
ax.add_artist(arrow_rew)
ax.text(6.1, (core_y+agent_y)/2, 'Reward\n(scalar)', 
        ha='left', va='center', fontsize=9, color='#F39C12', fontweight='bold')

# ============================================
# Layer 5: Training & Evaluation
# ============================================
train_y = 8.0
train_items = [
    ('Training\nLoop', 2, '#9B59B6'),
    ('Model\nCheckpoints', 4, '#34495E'),
    ('Evaluation\n& Metrics', 6, '#16A085'),
    ('Baseline\nComparison', 8, '#D35400')
]

for label, x, color in train_items:
    rect = FancyBboxPatch((x-0.6, train_y-0.4), 1.2, 0.8,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x, train_y, label, ha='center', va='center', 
            fontsize=9, fontweight='bold', color='white')

# Arrows from agent to training components
for label, x, color in train_items:
    arrow = FancyArrowPatch((5, agent_y+0.5), (x, train_y-0.45),
                           arrowstyle='->', mutation_scale=15, 
                           linewidth=1.5, color='gray', alpha=0.6)
    ax.add_artist(arrow)

# ============================================
# Side Panel: Key Features
# ============================================
features_x = 0.3
features_y = 6.5
features_text = """
KEY FEATURES:
• Multi-objective reward
• Battery degradation
• EV charging optimization
• Safety constraints
• Explainable decisions
• Real data integration
"""
ax.text(features_x, features_y, features_text, 
        ha='left', va='center', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='lightyellow', 
                 edgecolor='black', linewidth=2, pad=0.5))

# ============================================
# Side Panel: Action Space
# ============================================
action_x = 9.7
action_y = 6.5
action_text = """
ACTIONS:
1. Battery 1 Power
2. Battery 2 Power
3. Grid Import/Export
4. EV Charging
5. Curtailment
"""
ax.text(action_x, action_y, action_text, 
        ha='right', va='center', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='lightblue', 
                 edgecolor='black', linewidth=2, pad=0.5))

# ============================================
# Legend
# ============================================
legend_y = 0.2
legend_items = [
    ('Data Flow', 'gray', '-', 1.5),
    ('Observation', '#2ECC71', '-', 3),
    ('Action', '#E74C3C', '-', 3),
    ('Reward', '#F39C12', '-', 3)
]

for i, (label, color, style, width) in enumerate(legend_items):
    x_start = 3 + i*1.5
    ax.plot([x_start, x_start+0.5], [legend_y, legend_y], 
           color=color, linestyle=style, linewidth=width)
    ax.text(x_start+0.6, legend_y, label, ha='left', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('system_architecture.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✓ Saved system architecture diagram to: system_architecture.png")
plt.close()

# ============================================
# Create Flow Diagram
# ============================================
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'RL Training Flow', 
        ha='center', va='top', fontsize=18, fontweight='bold')

steps = [
    (5, 8.5, 'Initialize Environment\n& Agent', '#E8F4F8'),
    (5, 7.5, 'Reset Episode\n(t=0)', '#D4E6F1'),
    (5, 6.5, 'Get Observation\nobs_t', '#AED6F1'),
    (5, 5.5, 'Agent Selects Action\na_t = π(obs_t)', '#85C1E2'),
    (5, 4.5, 'Safety Check\nClip unsafe actions', '#FF6B6B'),
    (5, 3.5, 'Execute Action\nUpdate system state', '#76D7C4'),
    (5, 2.5, 'Calculate Reward\nr_t = -(cost+emissions+...)', '#F8C471'),
    (2, 1.5, 'Done?\n(t=96)', '#DDA0DD'),
    (8, 1.5, 'Store Transition\n(obs,a,r,obs_next)', '#B8E994'),
    (5, 0.5, 'Update Agent\n(PPO loss)', '#6C5CE7')
]

for x, y, text, color in steps:
    width = 2.5
    height = 0.6
    rect = FancyBboxPatch((x-width/2, y-height/2), width, height,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

# Add flow arrows
flow_arrows = [
    (5, 8.2, 5, 7.8),
    (5, 7.2, 5, 6.8),
    (5, 6.2, 5, 5.8),
    (5, 5.2, 5, 4.8),
    (5, 4.2, 5, 3.8),
    (5, 3.2, 5, 2.8),
    (5, 2.2, 3.25, 1.8),  # To "Done?"
    (3.25, 1.2, 5, 0.8),  # From "Done?" to "Update"
    (6.75, 1.5, 8, 1.5),  # "Done?" to "Store"
    (8, 1.2, 6.25, 6.2),  # "Store" back to "Get Obs"
]

for x1, y1, x2, y2 in flow_arrows:
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2, color='black')
    ax.add_artist(arrow)

# Labels for decision branches
ax.text(6.25, 1.7, 'Yes', ha='center', va='bottom', fontsize=9, color='green', fontweight='bold')
ax.text(4.5, 1.7, 'No', ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
ax.text(7.2, 4, 'Continue\nEpisode', ha='center', va='center', 
        fontsize=8, style='italic', color='blue')

plt.tight_layout()
plt.savefig('training_flow.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✓ Saved training flow diagram to: training_flow.png")
plt.close()

print("\n✓ Architecture diagrams created successfully!")
print("  - system_architecture.png: Shows all components and interactions")
print("  - training_flow.png: Shows the RL training loop")

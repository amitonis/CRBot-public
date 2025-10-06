import numpy as np
import time
import os
import pyautogui
import threading
from dotenv import load_dotenv
from Actions import Actions
from inference_sdk import InferenceHTTPClient

# Load environment variables from .env file
load_dotenv()

MAX_ENEMIES = 10
MAX_ALLIES = 10

SPELL_CARDS = ["Fireball", "Zap", "Arrows", "Tornado", "Rocket", "Lightning", "Freeze"]

# Strategic Card Archetypes
CARD_ARCHETYPES = {
    # Win Conditions (Building Targeting)
    "WIN_CONDITIONS": ["Giant", "Golem", "Lava Hound", "X-Bow", "Mortar", "Royal Giant", "Balloon", "Hog Rider", "Ram Rider", "Battle Ram", "Graveyard"],
    
    # Tanks
    "TANKS": ["Giant", "Golem", "Lava Hound", "P.E.K.K.A", "Mega Knight", "Royal Giant", "Giant Skeleton"],
    
    # Mini Tanks  
    "MINI_TANKS": ["Knight", "Valkyrie", "Mini P.E.K.K.A", "Prince", "Dark Prince", "Lumberjack", "Bandit", "Ram Rider"],
    
    # Swarm Cards
    "SWARM": ["Skeleton Army", "Minion Horde", "Goblin Gang", "Barbarians", "Elite Barbarians", "Three Musketeers", "Minions", "Goblins", "Spear Goblins"],
    
    # Anti-Air
    "ANTI_AIR": ["Musketeer", "Archers", "Minions", "Minion Horde", "Tesla", "Inferno Tower", "Wizard", "Executioner", "Hunter"],
    
    # Buildings
    "BUILDINGS": ["Tesla", "Inferno Tower", "Cannon", "Bomb Tower", "X-Bow", "Mortar", "Elixir Collector", "Tombstone", "Furnace", "Goblin Hut"],
    
    # Spells
    "SPELLS": ["Fireball", "Zap", "Arrows", "Lightning", "Rocket", "Log", "Freeze", "Tornado", "Poison", "Rage", "Mirror", "Clone"]
}

# Elixir Management Constants
MAX_ELIXIR = 10
ELIXIR_LEAK_THRESHOLD = 9  # Start playing cards when at 9+ elixir
ELIXIR_EFFICIENCY_THRESHOLD = 6  # Maintain at least 6 elixir for defense

# Spell cards for penalty logic
SPELL_CARDS = {
    "Arrows", "Fireball", "Zap", "Lightning", "Rocket", "Freeze", 
    "Poison", "Log", "Tornado", "Clone", "Rage", "Mirror", 
    "Snowball", "Barbarian Barrel", "Earthquake", "Graveyard",
    "Heal Spirit", "Royal Delivery"
}

class ClashRoyaleEnv:
    def __init__(self):
        self.actions = Actions()
        self.rf_model = self.setup_roboflow()
        self.card_model = self.setup_card_roboflow()
        self.state_size = 1 + 2 * (MAX_ALLIES + MAX_ENEMIES)

        self.num_cards = 4
        self.grid_width = 18
        self.grid_height = 28

        self.screenshot_path = os.path.join(os.path.dirname(__file__), 'screenshots', "current.png")
        self.available_actions = self.get_available_actions()
        self.action_size = len(self.available_actions)
        self.current_cards = []
        
        # Strategic State Tracking
        self.current_elixir = 5  # Starting elixir
        self.opponent_elixir_estimate = 5
        self.our_princess_towers = 2
        self.enemy_princess_towers = 2
        self.last_elixir_check = time.time()
        self.game_phase = "EARLY"  # EARLY, MID, LATE
        self.opponent_archetype = "UNKNOWN"
        self.opponent_cards_seen = []
        self.last_opponent_play = None
        self.elixir_leak_time = 0

        self.game_over_flag = None
        self._endgame_thread = None
        self._endgame_thread_stop = threading.Event()

        self.prev_elixir = None
        self.prev_enemy_presence = None

        self.prev_enemy_princess_towers = None

        self.match_over_detected = False
    
    def update_elixir_management(self):
        """Update elixir tracking and detect leaking"""
        current_time = time.time()
        time_since_last_check = current_time - self.last_elixir_check
        
        # Get actual elixir from game
        actual_elixir = self.actions.count_elixir()
        if actual_elixir > 0:
            self.current_elixir = actual_elixir
        
        # Track elixir leak time (above 9 elixir)
        if self.current_elixir >= ELIXIR_LEAK_THRESHOLD:
            self.elixir_leak_time += time_since_last_check
        else:
            self.elixir_leak_time = 0
            
        self.last_elixir_check = current_time
    
    def detect_card_archetype(self, card_name):
        """Determine the archetype of a card"""
        for archetype, cards in CARD_ARCHETYPES.items():
            if card_name in cards:
                return archetype
        return "UNKNOWN"
    
    def analyze_opponent_troops(self, enemy_troops):
        """Analyze opponent's troops and update strategy"""
        if not enemy_troops:
            return
            
        # Count different troop types
        tank_count = 0
        swarm_count = 0
        win_condition_count = 0
        
        for troop_pos in enemy_troops:
            # For now, we can't identify specific troop types from positions
            # But we can analyze positioning patterns
            x, y = troop_pos['x'], troop_pos['y']
            
            # Troops near our towers (defensive threat)
            if y > 0.7:
                tank_count += 1  # Assume tanks push to towers
            # Troops spread out (swarm pattern)
            elif len(enemy_troops) > 3:
                swarm_count += 1
        
        # Update opponent archetype based on patterns
        if tank_count > swarm_count:
            self.opponent_archetype = "TANKS"
        elif swarm_count > 1:
            self.opponent_archetype = "SWARM"
        else:
            self.opponent_archetype = "BALANCED"
    
    def should_defend(self, enemy_troops):
        """Determine if we should prioritize defense"""
        if not enemy_troops:
            return False
        
        # Count enemies near our towers (y > 0.6 = approaching our side)
        threats_near_towers = sum(1 for troop in enemy_troops if troop['y'] > 0.6)
        
        # Defend if multiple threats or single threat very close
        return threats_near_towers >= 2 or any(troop['y'] > 0.8 for troop in enemy_troops)
    
    def should_counter_push(self, our_troops, enemy_troops):
        """Determine if we should counter-push"""
        if not our_troops:
            return False
            
        # Counter-push if we have troops advancing and enemy has few defenders
        our_advancing = sum(1 for troop in our_troops if troop['y'] < 0.4)  # Our troops on enemy side
        enemy_defending = sum(1 for troop in enemy_troops if troop['y'] < 0.5)  # Enemy troops defending
        
        return our_advancing > 0 and enemy_defending <= 1
    
    def should_pressure(self, current_elixir, enemy_troops):
        """Determine if we should apply pressure"""
        # Pressure if we have elixir advantage and no immediate threats
        elixir_advantage = self.current_elixir - self.opponent_elixir_estimate
        immediate_threats = sum(1 for troop in enemy_troops if troop['y'] > 0.7)
        
        return (elixir_advantage > 2 and 
                immediate_threats == 0 and 
                current_elixir >= ELIXIR_EFFICIENCY_THRESHOLD)
    
    def get_strategic_decision(self, our_troops, enemy_troops):
        """Make strategic decision based on game state"""
        # Update elixir management first
        self.update_elixir_management()
        
        # Analyze opponent troops
        self.analyze_opponent_troops(enemy_troops)
        
        # Priority 1: Prevent elixir leak
        if self.current_elixir >= ELIXIR_LEAK_THRESHOLD:
            return "PREVENT_LEAK"
        
        # Priority 2: Defend against threats
        elif self.should_defend(enemy_troops):
            return "DEFEND"
        
        # Priority 3: Counter-push opportunities
        elif self.should_counter_push(our_troops, enemy_troops):
            return "COUNTER_PUSH"
        
        # Priority 4: Apply pressure
        elif self.should_pressure(self.current_elixir, enemy_troops):
            return "PRESSURE"
        
        # Default: Wait for better opportunity
        else:
            return "WAIT"
    
    def get_strategic_positions(self, card_name=None):
        """Get 6-8 strategic positions based on tower status and card type"""
        positions = []
        
        # Base positions based on princess tower status
        if self.our_princess_towers == 2 and self.enemy_princess_towers == 2:
            # Both sides have towers - standard 8 positions
            base_positions = [
                (0.3, 0.7), (0.7, 0.7),  # Behind towers
                (0.2, 0.5), (0.8, 0.5),  # Bridge positions
                (0.4, 0.6), (0.6, 0.6),  # Support positions
                (0.5, 0.8), (0.5, 0.4)   # Center defensive/offensive
            ]
        elif self.our_princess_towers == 1:
            # We lost a tower - more defensive positions
            base_positions = [
                (0.5, 0.8), (0.5, 0.7),  # Center defensive
                (0.4, 0.6), (0.6, 0.6),  # Support positions
                (0.3, 0.5), (0.7, 0.5),  # Careful bridge
                (0.5, 0.9), (0.4, 0.8)   # Deep defense
            ]
        elif self.enemy_princess_towers == 1:
            # Enemy lost tower - more aggressive positions
            base_positions = [
                (0.2, 0.4), (0.8, 0.4),  # Aggressive bridge
                (0.3, 0.3), (0.7, 0.3),  # Enemy side pressure
                (0.5, 0.4), (0.5, 0.3),  # Center pressure
                (0.4, 0.5), (0.6, 0.5)   # Support positions
            ]
        elif self.our_princess_towers == 0 or self.enemy_princess_towers == 0:
            # King tower only - focus center
            base_positions = [
                (0.5, 0.7), (0.5, 0.6),  # Center focus
                (0.4, 0.5), (0.6, 0.5),  # Balanced positions
                (0.3, 0.4), (0.7, 0.4),  # Wide positions
                (0.5, 0.8), (0.5, 0.3)   # Defensive/offensive
            ]
        else:
            # Default positions
            base_positions = [
                (0.4, 0.7), (0.6, 0.7),  # Standard defensive
                (0.2, 0.5), (0.8, 0.5),  # Bridge
                (0.5, 0.6), (0.5, 0.4),  # Center
                (0.3, 0.6), (0.7, 0.6)   # Support
            ]
        
        # Adjust positions based on card archetype
        if card_name:
            card_archetype = self.detect_card_archetype(card_name)
            
            if card_archetype == "TANKS":
                # Tanks - behind towers or bridge for push
                positions = [(0.2, 0.6), (0.8, 0.6), (0.5, 0.7), (0.3, 0.5), (0.7, 0.5), (0.4, 0.7)]
            elif card_archetype == "WIN_CONDITIONS":
                # Win conditions - bridge positions and protected spots
                positions = [(0.2, 0.5), (0.8, 0.5), (0.3, 0.6), (0.7, 0.6), (0.5, 0.5), (0.4, 0.4)]
            elif card_archetype == "SWARM":
                # Swarm - defensive and counter positions
                positions = [(0.4, 0.7), (0.6, 0.7), (0.3, 0.6), (0.7, 0.6), (0.5, 0.8), (0.2, 0.6)]
            elif card_archetype == "BUILDINGS":
                # Buildings - defensive positions only
                positions = [(0.4, 0.8), (0.6, 0.8), (0.5, 0.75), (0.3, 0.8), (0.7, 0.8), (0.5, 0.9)]
            elif card_archetype == "SPELLS":
                # Spells - enemy positions for damage
                positions = [(0.3, 0.3), (0.7, 0.3), (0.5, 0.2), (0.4, 0.4), (0.6, 0.4), (0.5, 0.35)]
            else:
                # Default positions
                positions = base_positions[:6]
        else:
            positions = base_positions
        
        # Ensure we return 6-8 positions as requested
        return positions[:8]

    def _parse_troops_from_state(self, state):
        """Parse troops from state vector for testing purposes"""
        our_troops = []
        enemy_troops = []
        
        # Parse ally troops (after elixir)
        ally_start = 1
        for i in range(ally_start, ally_start + 2 * MAX_ALLIES, 2):
            if i + 1 < len(state):
                x, y = state[i], state[i + 1]
                if x != 0.0 or y != 0.0:
                    our_troops.append({'x': x, 'y': y, 'class': 'ally_troop'})
        
        # Parse enemy troops
        enemy_start = 1 + 2 * MAX_ALLIES
        for i in range(enemy_start, enemy_start + 2 * MAX_ENEMIES, 2):
            if i + 1 < len(state):
                x, y = state[i], state[i + 1]
                if x != 0.0 or y != 0.0:
                    enemy_troops.append({'x': x, 'y': y, 'class': 'enemy_troop'})
        
        return our_troops, enemy_troops

    def _count_our_princess_towers(self):
        """Count our princess towers (for testing, assume 2)"""
        return 2

    def setup_roboflow(self):
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable is not set. Please check your .env file.")
        
        return InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=api_key
        )

    def setup_card_roboflow(self):
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable is not set. Please check your .env file.")
        
        return InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=api_key
        )

    def reset(self):
        # self.actions.click_battle_start()
        # Instead, just wait for the new game to load after clicking "Play Again"
        time.sleep(3)
        self.game_over_flag = None
        self._endgame_thread_stop.clear()
        self._endgame_thread = threading.Thread(target=self._endgame_watcher, daemon=True)
        self._endgame_thread.start()
        self.prev_elixir = None
        self.prev_enemy_presence = None
        self.prev_enemy_princess_towers = self._count_enemy_princess_towers()
        self.match_over_detected = False
        return self._get_state()

    def close(self):
        self._endgame_thread_stop.set()
        if self._endgame_thread:
            self._endgame_thread.join()

    def step(self, action_index):
        # Check for match over
        if not self.match_over_detected and hasattr(self.actions, "detect_match_over") and self.actions.detect_match_over():
            print("Match over detected (matchover.png), forcing no-op until next game.")
            self.match_over_detected = True

        # If match over, only allow no-op action (last action in list)
        if self.match_over_detected:
            action_index = len(self.available_actions) - 1  # No-op action

        if self.game_over_flag:
            done = True
            reward = self._compute_reward(self._get_state())
            result = self.game_over_flag
            if result == "victory":
                reward += 100
                print("Victory detected - ending episode")
            elif result == "defeat":
                reward -= 100
                print("Defeat detected - ending episode")
            self.match_over_detected = False  # Reset for next episode
            return self._get_state(), reward, done

        # Get current game state
        current_state = self._get_state()
        if current_state is None:
            print("Failed to get game state, skipping turn")
            return self._get_state(), 0, False

        # Update elixir management
        current_elixir = current_state[0] * 10
        self.update_elixir_management()

        self.current_cards = self.detect_cards_in_hand()
        print("\nCurrent cards in hand:", self.current_cards)

        # If all cards are "Unknown", click at (1611, 831) and return no-op
        if all(card == "Unknown" for card in self.current_cards):
            print("All cards are Unknown, clicking at (1611, 831) and skipping move.")
            pyautogui.moveTo(1611, 831, duration=0.2)
            pyautogui.click()
            # Return current state, zero reward, not done
            next_state = self._get_state()
            return next_state, 0, False

        # Parse opponent troops from current state for strategic analysis
        opponent_troops = []
        # Extract enemy positions from state (after elixir and ally positions)
        enemy_start_idx = 1 + 2 * MAX_ALLIES
        for i in range(enemy_start_idx, enemy_start_idx + 2 * MAX_ENEMIES, 2):
            if i + 1 < len(current_state):
                ex = current_state[i]
                ey = current_state[i + 1]
                if ex != 0.0 or ey != 0.0:
                    # Convert normalized coords back to pixel coords for analysis
                    ex_px = int(ex * self.actions.WIDTH)
                    ey_px = int(ey * self.actions.HEIGHT)
                    opponent_troops.append({
                        'x': ex_px, 
                        'y': ey_px,
                        'class': 'enemy_troop'  # Generic class since we don't have specific troop types
                    })

        # Analyze opponent troops
        self.analyze_opponent_troops(opponent_troops)

        # Get strategic decision
        # Parse our troops for strategic decision
        our_troops = []
        ally_start_idx = 1
        for i in range(ally_start_idx, ally_start_idx + 2 * MAX_ALLIES, 2):
            if i + 1 < len(current_state):
                ax = current_state[i]
                ay = current_state[i + 1]
                if ax != 0.0 or ay != 0.0:
                    ax_px = int(ax * self.actions.WIDTH)
                    ay_px = int(ay * self.actions.HEIGHT)
                    our_troops.append({
                        'x': ax_px, 
                        'y': ay_px,
                        'class': 'ally_troop'
                    })
        
        strategic_decision = self.get_strategic_decision(our_troops, opponent_troops)
        print(f"Strategic decision: {strategic_decision}")

        # Use strategic decision to override action if needed
        action = self.available_actions[action_index]
        card_index, x_frac, y_frac = action

        # Apply strategic positioning if we're playing a card
        if card_index != -1 and card_index < len(self.current_cards):
            card_name = self.current_cards[card_index]
            card_archetype = self.detect_card_archetype(card_name)
            
            # Get strategic positions (6-8 positions based on towers)
            strategic_positions = self.get_strategic_positions(card_archetype)
            
            # Choose best strategic position based on decision
            if strategic_positions:
                if strategic_decision == "defend" and opponent_troops:
                    # Find position closest to strongest threat
                    threat_y = max(troop['y'] for troop in opponent_troops)
                    best_pos = min(strategic_positions, key=lambda pos: abs(pos[1] - threat_y))
                elif strategic_decision == "counter_push":
                    # Use offensive positions (lower y values)
                    offensive_positions = [pos for pos in strategic_positions if pos[1] < self.actions.HEIGHT * 0.5]
                    best_pos = offensive_positions[0] if offensive_positions else strategic_positions[0]
                elif strategic_decision == "pressure":
                    # Use bridge positions for pressure
                    bridge_positions = [pos for pos in strategic_positions if abs(pos[1] - self.actions.HEIGHT * 0.5) < 50]
                    best_pos = bridge_positions[0] if bridge_positions else strategic_positions[0]
                else:
                    # Default to first strategic position
                    best_pos = strategic_positions[0]
                
                # Strategic positions are already normalized coordinates
                x_frac = best_pos[0]
                y_frac = best_pos[1]
                print(f"Using strategic position: ({best_pos[0]:.2f}, {best_pos[1]:.2f}) -> ({x_frac:.2f}, {y_frac:.2f})")

        print(f"Action selected: card_index={card_index}, x_frac={x_frac:.2f}, y_frac={y_frac:.2f}")

        spell_penalty = 0

        if card_index != -1 and card_index < len(self.current_cards):
            card_name = self.current_cards[card_index]
            print(f"Attempting to play {card_name}")
            x = int(x_frac * self.actions.WIDTH) + self.actions.TOP_LEFT_X
            y = int(y_frac * self.actions.HEIGHT) + self.actions.TOP_LEFT_Y
            self.actions.card_play(x, y, card_index)
            time.sleep(1)  # You can reduce this if needed

            # --- Enhanced spell penalty logic ---
            if card_name in SPELL_CARDS:
                # Use opponent_troops for more accurate spell targeting
                enemy_positions = [(troop['x'], troop['y']) for troop in opponent_troops]
                radius = 100
                found_enemy = any((abs(ex - x) ** 2 + abs(ey - y) ** 2) ** 0.5 < radius for ex, ey in enemy_positions)
                if not found_enemy:
                    spell_penalty = -5  # Penalize for wasting spell

        # --- Princess tower reward logic ---
        current_enemy_princess_towers = self._count_enemy_princess_towers()
        princess_tower_reward = 0
        if self.prev_enemy_princess_towers is not None:
            if current_enemy_princess_towers < self.prev_enemy_princess_towers:
                princess_tower_reward = 20
        self.prev_enemy_princess_towers = current_enemy_princess_towers

        done = False
        reward = self._compute_reward(self._get_state()) + spell_penalty + princess_tower_reward
        next_state = self._get_state()
        return next_state, reward, done

    def _get_state(self):
        self.actions.capture_area(self.screenshot_path)
        elixir = self.actions.count_elixir()
        
        workspace_name = os.getenv('WORKSPACE_TROOP_DETECTION')
        if not workspace_name:
            raise ValueError("WORKSPACE_TROOP_DETECTION environment variable is not set. Please check your .env file.")
        
        results = self.rf_model.run_workflow(
            workspace_name=workspace_name,
            workflow_id="custom-workflow-4",
            images={"image": self.screenshot_path},
            use_cache=True
        )

        print("RAW results:", results)

        # Handle new structure: dict with "predictions" key
        predictions = []
        if isinstance(results, dict) and "predictions" in results:
            predictions = results["predictions"]
        elif isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict) and "predictions" in first:
                predictions = first["predictions"]
        print("Predictions:", predictions)
        if not predictions:
            print("WARNING: No predictions found in results")
            return None

        # After getting 'predictions' from results:
        if isinstance(predictions, dict) and "predictions" in predictions:
            predictions = predictions["predictions"]

        print("RAW predictions:", predictions)
        print("Detected classes:", [repr(p.get("class", "")) for p in predictions if isinstance(p, dict)])

        TOWER_CLASSES = {
            "ally king tower",
            "ally princess tower",
            "enemy king tower",
            "enemy princess tower"
        }

        def normalize_class(cls):
            return cls.strip().lower() if isinstance(cls, str) else ""

        allies = [
            (p["x"], p["y"])
            for p in predictions
            if (
                isinstance(p, dict)
                and normalize_class(p.get("class", "")) not in TOWER_CLASSES
                and normalize_class(p.get("class", "")).startswith("ally")
                and "x" in p and "y" in p
            )
        ]

        enemies = [
            (p["x"], p["y"])
            for p in predictions
            if (
                isinstance(p, dict)
                and normalize_class(p.get("class", "")) not in TOWER_CLASSES
                and normalize_class(p.get("class", "")).startswith("enemy")
                and "x" in p and "y" in p
            )
        ]

        print("Allies:", allies)
        print("Enemies:", enemies)

        # Normalize positions
        def normalize(units):
            return [(x / self.actions.WIDTH, y / self.actions.HEIGHT) for x, y in units]

        # Pad or truncate to fixed length
        def pad_units(units, max_units):
            units = normalize(units)
            if len(units) < max_units:
                units += [(0.0, 0.0)] * (max_units - len(units))
            return units[:max_units]

        ally_positions = pad_units(allies, MAX_ALLIES)
        enemy_positions = pad_units(enemies, MAX_ENEMIES)

        # Flatten positions
        ally_flat = [coord for pos in ally_positions for coord in pos]
        enemy_flat = [coord for pos in enemy_positions for coord in pos]

        state = np.array([elixir / 10.0] + ally_flat + enemy_flat, dtype=np.float32)
        return state

    def _compute_reward(self, state):
        if state is None:
            return 0

        elixir = state[0] * 10

        # Sum all enemy positions (not just the first)
        enemy_positions = state[1 + 2 * MAX_ALLIES:]  # All enemy x1, y1, x2, y2, ...
        # enemy_presence = sum(enemy_positions)
        enemy_presence = sum(enemy_positions[1::2]) # Only y coords so it does not bias left/right side
        reward = -enemy_presence

        # Elixir efficiency: reward for spending elixir if it reduces enemy presence
        if self.prev_elixir is not None and self.prev_enemy_presence is not None:
            elixir_spent = self.prev_elixir - elixir
            enemy_reduced = self.prev_enemy_presence - enemy_presence
            if elixir_spent > 0 and enemy_reduced > 0:
                reward += 2 * min(elixir_spent, enemy_reduced)  # tune this factor

        self.prev_elixir = elixir
        self.prev_enemy_presence = enemy_presence

        return reward

    def detect_cards_in_hand(self):
        try:
            card_paths = self.actions.capture_individual_cards()
            print("\nTesting individual card predictions:")

            cards = []
            workspace_name = os.getenv('WORKSPACE_CARD_DETECTION')
            if not workspace_name:
                raise ValueError("WORKSPACE_CARD_DETECTION environment variable is not set. Please check your .env file.")
            
            for card_path in card_paths:
                results = self.card_model.run_workflow(
                    workspace_name=workspace_name,
                    workflow_id="custom-workflow-4",
                    images={"image": card_path},
                    use_cache=True
                )
                # print("Card detection raw results:", results)  # Debug print

                # Fix: parse nested structure
                predictions = []
                if isinstance(results, list) and results:
                    preds_dict = results[0].get("predictions", {})
                    if isinstance(preds_dict, dict):
                        predictions = preds_dict.get("predictions", [])
                if predictions:
                    card_name = predictions[0]["class"]
                    print(f"Detected card: {card_name}")
                    cards.append(card_name)
                else:
                    print("No card detected.")
                    cards.append("Unknown")
            return cards
        except Exception as e:
            print(f"Error in detect_cards_in_hand: {e}")
            return []

    def get_available_actions(self):
        """Generate all possible actions"""
        actions = [
            [card, x / (self.grid_width - 1), y / (self.grid_height - 1)]
            for card in range(self.num_cards)
            for x in range(self.grid_width)
            for y in range(self.grid_height)
        ]
        actions.append([-1, 0, 0])  # No-op action
        return actions

    def _endgame_watcher(self):
        while not self._endgame_thread_stop.is_set():
            result = self.actions.detect_game_end()
            if result:
                self.game_over_flag = result
                break
            # Sleep a bit to avoid hammering the CPU
            time.sleep(0.5)

    def _count_enemy_princess_towers(self):
        self.actions.capture_area(self.screenshot_path)
        
        workspace_name = os.getenv('WORKSPACE_TROOP_DETECTION')
        if not workspace_name:
            raise ValueError("WORKSPACE_TROOP_DETECTION environment variable is not set. Please check your .env file.")
        
        results = self.rf_model.run_workflow(
            workspace_name=workspace_name,
            workflow_id="custom-workflow-4",
            images={"image": self.screenshot_path},
            use_cache=True
        )
        predictions = []
        if isinstance(results, dict) and "predictions" in results:
            predictions = results["predictions"]
        elif isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict) and "predictions" in first:
                predictions = first["predictions"]
        return sum(1 for p in predictions if isinstance(p, dict) and p.get("class") == "enemy princess tower")
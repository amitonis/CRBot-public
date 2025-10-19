import numpy as np
import time
import os
import pyautogui
import threading
import random
from dotenv import load_dotenv
from Actions import Actions
from inference_sdk import InferenceHTTPClient

load_dotenv()

MAX_ENEMIES = 10
MAX_ALLIES = 10

SPELL_CARDS = ["Fireball", "Zap", "Arrows", "Tornado", "Rocket", "Lightning", "Freeze"]

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

# Card Costs for Elixir Management
CARD_COSTS = {
    # 1 Elixir
    "Skeletons": 1, "Ice Spirit": 1, "Heal Spirit": 1,
    
    # 2 Elixir  
    "Zap": 2, "Ice Golem": 2, "Bats": 2, "Spear Goblins": 2, "Goblins": 2, "Archers": 2, "Bomber": 2, "Arrows": 3,
    
    # 3 Elixir
    "Knight": 3, "Archers": 3, "Minions": 3, "Tesla": 4, "Cannon": 3, "Goblin Gang": 3, "Princess": 3, "Miner": 3, "Log": 2,
    
    # 4 Elixir
    "Fireball": 4, "Musketeer": 4, "Mini P.E.K.K.A": 4, "Hog Rider": 4, "Valkyrie": 4, "Baby Dragon": 4, "Dark Prince": 4, "Battle Ram": 4, "Inferno Tower": 5, "Tombstone": 3,
    
    # 5 Elixir
    "Prince": 5, "Giant": 5, "Wizard": 5, "Witch": 5, "Balloon": 5, "Inferno Dragon": 4, "Bowler": 5, "Executioner": 5, "Bandit": 3, "Electro Wizard": 4,
    
    # 6 Elixir
    "Lightning": 6, "Rocket": 6, "P.E.K.K.A": 7, "Giant Skeleton": 6, "Elite Barbarians": 6, "Sparky": 6, "Lumberjack": 4, "Miner": 3, "Graveyard": 5,
    
    # 7+ Elixir
    "Golem": 8, "Lava Hound": 7, "Mega Knight": 7, "Three Musketeers": 9, "Royal Giant": 6
}

class ClashRoyaleEnv:
    def __init__(self):
        self.actions = Actions()
        self.rf_model = self.setup_roboflow()
        self.card_model = self.setup_card_roboflow()
        self.base_state_size = 1 + 2 * (MAX_ALLIES + MAX_ENEMIES)

        self.num_cards = 4
        self.grid_width = 18
        self.grid_height = 28

        self.screenshot_path = os.path.join(os.path.dirname(__file__), 'screenshots', "current.png")
        self.available_actions = self.get_available_actions()
        self.action_size = len(self.available_actions)
        self.current_cards = []
        self.latest_enemy_entities = []
        self.latest_ally_entities = []
        
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
        self.elixir_urgency = "LOW"  # CRITICAL, HIGH, MEDIUM, LOW, SAVE
        
        # 🧠 OPPONENT RESPONSE TRACKING
        self.opponent_memory = {
            "last_cards": [],  # Track opponent's last 10 cards
            "responses_to_my_cards": {},  # What opponent plays after my cards
            "counter_patterns": {},  # Opponent's preferred counters
            "preferred_lanes": {"left": 0, "right": 0, "center": 0},  # Lane preferences
            "reaction_time": [],  # How fast opponent responds
            "elixir_usage_pattern": []  # Opponent's elixir spending habits
        }
        
        # 📍 DYNAMIC POSITION LEARNING
        self.position_memory = {
            "success_heatmap": {},  # Track success rate for each position
            "card_position_success": {},  # Success rate by card type and position
            "recent_failures": [],  # Last 5 failed positions to avoid
            "opponent_weakness_map": {"left": 0.5, "right": 0.5, "center": 0.5},  # Detected weaknesses
            "last_successful_positions": [],  # Keep track of what worked
            "position_variety_counter": {}  # Ensure we don't repeat same positions
        }
        
        # 🎯 ADAPTIVE BEHAVIOR TRACKING
        self.my_last_card = None
        self.my_last_position = None
        self.my_last_play_time = time.time()
        self.cards_played_this_game = []
        self.positions_used_this_game = []
        
        # 🔄 META-GAME STATE RECOGNITION
        self.game_state = {
            "elixir_advantage": 0,  # Estimated elixir advantage over opponent
            "momentum": "NEUTRAL",  # WINNING, LOSING, NEUTRAL
            "game_phase": "EARLY",  # EARLY (0-1min), MID (1-2min), LATE (2min+), OVERTIME
            "pressure_level": "NONE",  # HIGH, MEDIUM, LOW, NONE
            "tower_advantage": 0,  # +1 if we destroyed tower, -1 if opponent did
            "last_momentum_change": time.time(),
            "game_start_time": time.time()
        }
        
        # ⚡ CONTEXTUAL ACTION SELECTION
        self.action_context = {
            "mode": "BALANCED",  # REACTIVE, PROACTIVE, DEFENSIVE, AGGRESSIVE, BALANCED
            "last_opponent_aggression": time.time(),
            "counter_window": False,  # Opportunity to counter-attack
            "bait_opportunity": False,  # Opportunity to bait opponent's counters
            "cycle_priority": None,  # Card we're trying to cycle to
            "forced_defense": False  # Must defend immediately
        }
        
        # 🎯 MULTI-OBJECTIVE LEARNING
        self.objectives = {
            "damage_dealt": 0,
            "damage_taken": 0, 
            "elixir_efficiency": [],  # Track elixir trades
            "successful_defenses": 0,
            "successful_attacks": 0,
            "adaptation_score": 0,  # How well we adapt to opponent
            "position_variety_score": 0,  # Bonus for using different positions
            "timing_score": 0  # Bonus for good timing
        }
        
        # 🎲 RANDOMIZED STRATEGY PATTERNS
        self.strategy_patterns = {
            "current_pattern": "ADAPTIVE",  # AGGRESSIVE, DEFENSIVE, CYCLE, ADAPTIVE, RANDOM
            "pattern_duration": 0,  # How long we've used current pattern
            "pattern_success": [],  # Success rate of recent patterns
            "randomization_level": 0.3,  # How much randomness to inject (0.0-1.0)
            "last_pattern_change": time.time(),
            "deception_mode": False,  # Are we intentionally playing suboptimally?
            "bluff_cards": []  # Cards we're pretending to not have
        }
        
        # 🔮 PREDICTIVE OPPONENT MODELING
        self.opponent_model = {
            "play_patterns": {},  # {"card_sequence": count}
            "timing_patterns": [],  # When opponent typically plays
            "elixir_patterns": [],  # Opponent's elixir management style
            "predicted_next_card": None,  # What we think they'll play next
            "prediction_confidence": 0.0,  # How confident we are
            "cycle_tracking": [],  # Their likely card cycle order
            "aggression_prediction": "NEUTRAL"  # AGGRESSIVE, DEFENSIVE, NEUTRAL
        }
        
        # 🎭 DECEPTION & BLUFFING SYSTEM
        self.deception = {
            "fake_weakness": None,  # Pretend to be weak to this card type
            "hidden_strength": None,  # Don't reveal this strong card early
            "bait_attempts": 0,  # How many times we've tried to bait
            "successful_baits": 0,  # How many baits worked
            "opponent_baited": [],  # What we've successfully baited
            "next_bait_target": None,  # What we're trying to bait next
            "misdirection_active": False  # Are we currently misdirecting?
        }

        self.counter_counter_map = {
            "inferno tower": ["zap", "electro wizard", "lightning"],
            "minion horde": ["arrows", "fireball", "baby dragon"],
            "barbarians": ["valkyrie", "fireball", "mega knight"],
            "tesla": ["lightning", "earthquake", "royal giant"],
            "tornado": ["electro wizard", "bowler", "royal ghost"],
            "rocket": ["royal giant", "hog rider", "miner"]
        }
        
        # ⚖️ RISK-REWARD ASSESSMENT
        self.risk_assessment = {
            "current_risk_level": "MEDIUM",  # LOW, MEDIUM, HIGH, EXTREME
            "risk_tolerance": 0.5,  # How much risk we're willing to take (0.0-1.0)
            "recent_risks": [],  # Track recent risky plays and outcomes
            "safe_plays_streak": 0,  # How many safe plays in a row
            "high_risk_cooldown": 0,  # Cooldown after failed high-risk play
            "calculated_risks": []  # Track risk calculations
        }
        
        # 🔄 COUNTER-META ADAPTATION
        self.meta_adaptation = {
            "detected_meta": "UNKNOWN",  # BEATDOWN, CYCLE, CONTROL, SIEGE, BRIDGE_SPAM
            "meta_confidence": 0.0,  # How sure we are about their meta
            "counter_strategy": "ADAPTIVE",  # Our counter-strategy
            "meta_weaknesses": [],  # Known weaknesses of their meta
            "adaptation_success": 0,  # How well our counter-strategy works
            "last_meta_detection": time.time()
        }
        
        # 🧩 COMBO CHAIN RECOGNITION
        self.combo_system = {
            "available_combos": {},  # {"combo_name": ["card1", "card2", ...]}
            "combo_success_rate": {},  # Track success of each combo
            "current_combo_setup": None,  # Combo we're currently setting up
            "combo_readiness": {},  # How ready each combo is (elixir/cards)
            "opponent_combo_threats": [],  # Combos opponent might have
            "combo_counter_ready": {}  # Our counters to opponent combos
        }
        
        # Initialize combo database
        self.initialize_combo_database()

        # 🏗️ HIERARCHICAL STRATEGY SELECTION
        self.strategy_hierarchy = {
            "high_level_strategy": "BALANCED",  # MACRO choices like PRESSURE, CONTROL, RUSH
            "recent_strategies": [],
            "strategy_success": {},
            "explore_probability": 0.2  # Probability to explore new high-level plans
        }

        # 🎲 ADVERSARIAL STRATEGY MIXING (Game Theory Inspired)
        self.strategy_mixing = {
            "mixed_strategy": {"ATTACK": 0.33, "DEFEND": 0.33, "PRESSURE": 0.34},
            "regret_minimization": {"ATTACK": 0.0, "DEFEND": 0.0, "PRESSURE": 0.0},
            "last_payoffs": [],
            "history_window": 10
        }

        # 📚 IMITATION & META-LEARNING PLACEHOLDERS
        self.imitation_memory = {
            "pro_patterns": [],  # Store sequences inspired by pro play
            "successful_sequences": []
        }

        # 🧠 MULTI-AGENT AWARENESS
        self.opponent_profiles = {
            "current_profile": "UNKNOWN",
            "profile_confidence": 0.0,
            "profile_history": []
        }

        # 🔢 ADVANCED STATE FEATURE TRACKING
        self.extra_state_features = [
            "elixir_advantage",
            "momentum",
            "game_phase",
            "pressure_level",
            "risk_tolerance",
            "randomization_level",
            "forced_defense",
            "opponent_aggression",
            "prediction_confidence",
            "avg_reaction_time",
            "lane_pref_left",
            "lane_pref_right",
            "meta_confidence",
            "combo_best_success"
        ]
        self.state_size = self.base_state_size + len(self.extra_state_features)

        self.game_over_flag = None
        self._endgame_thread = None
        self._endgame_thread_stop = threading.Event()

        self.prev_elixir = None
        self.prev_enemy_presence = None

        self.prev_enemy_princess_towers = None

        self.match_over_detected = False
    
    def update_elixir_management(self):
        """Advanced elixir tracking and leak prevention"""
        current_time = time.time()
        time_since_last_check = current_time - self.last_elixir_check
        
        # Get actual elixir from game
        actual_elixir = self.actions.count_elixir()
        if actual_elixir > 0:
            self.current_elixir = actual_elixir
        
        # Advanced leak detection with urgency levels
        if self.current_elixir >= 10:  # Full elixir - CRITICAL leak
            self.elixir_leak_time += time_since_last_check
            self.elixir_urgency = "CRITICAL"
        elif self.current_elixir >= 9:  # Near full - HIGH urgency
            self.elixir_leak_time += time_since_last_check * 0.5
            self.elixir_urgency = "HIGH"
        elif self.current_elixir >= 8:  # Getting high - MEDIUM urgency
            self.elixir_urgency = "MEDIUM"
        elif self.current_elixir >= 6:  # Good range - LOW urgency
            self.elixir_urgency = "LOW"
            self.elixir_leak_time = 0
        else:  # Low elixir - SAVE mode
            self.elixir_urgency = "SAVE"
            self.elixir_leak_time = 0
            
        self.last_elixir_check = current_time
    
    def get_card_cost(self, card_name):
        """Get elixir cost of a card"""
        return CARD_COSTS.get(card_name, 4)  # Default to 4 if unknown
    
    def get_playable_cards(self, available_cards):
        """Get cards that can be played with current elixir"""
        playable = []
        for card in available_cards:
            if self.get_card_cost(card) <= self.current_elixir:
                playable.append(card)
        return playable
    
    def get_best_elixir_card(self, available_cards, strategy="PREVENT_LEAK"):
        """Select best card based on elixir management strategy"""
        playable_cards = self.get_playable_cards(available_cards)
        if not playable_cards:
            return None
            
        if strategy == "PREVENT_LEAK":
            if self.elixir_urgency == "CRITICAL":
                # Play any card to prevent leak
                return max(playable_cards, key=lambda c: self.get_card_cost(c))
            elif self.elixir_urgency == "HIGH":
                # Prefer medium-cost cards (4-6 elixir)
                medium_cards = [c for c in playable_cards if 4 <= self.get_card_cost(c) <= 6]
                return medium_cards[0] if medium_cards else playable_cards[0]
            else:
                # Normal play - prefer efficient cards
                return min(playable_cards, key=lambda c: self.get_card_cost(c))
                
        elif strategy == "DEFEND":
            # Prefer defensive cards with good elixir efficiency
            defensive_cards = [c for c in playable_cards if self.detect_card_archetype(c) in ["BUILDINGS", "SWARM", "ANTI_AIR"]]
            if defensive_cards:
                return min(defensive_cards, key=lambda c: self.get_card_cost(c))
            return min(playable_cards, key=lambda c: self.get_card_cost(c))
            
        elif strategy == "COUNTER_PUSH":
            # Prefer win conditions and tanks when we have enough elixir
            if self.current_elixir >= 6:
                offensive_cards = [c for c in playable_cards if self.detect_card_archetype(c) in ["WIN_CONDITIONS", "TANKS"]]
                if offensive_cards:
                    return offensive_cards[0]
            return min(playable_cards, key=lambda c: self.get_card_cost(c))
            
        elif strategy == "PRESSURE":
            # Prefer cheap pressure cards
            cheap_cards = [c for c in playable_cards if self.get_card_cost(c) <= 4]
            return cheap_cards[0] if cheap_cards else playable_cards[0]
            
        return playable_cards[0] if playable_cards else None
    
    def track_opponent_response(self, opponent_troops):
        """Track opponent's response to our last play"""
        if not opponent_troops:
            return

        current_time = time.time()
        reaction_time = current_time - self.my_last_play_time if self.my_last_play_time else 0.0

        # Derive opponent card identifiers from troop metadata
        new_responses = []
        prev_count = len(self.opponent_memory.get("last_detected_troops", []))
        if len(opponent_troops) > prev_count:
            new_responses = opponent_troops[prev_count:]
        elif opponent_troops:
            new_responses = [opponent_troops[-1]]

        for troop in new_responses:
            opponent_card = troop.get('class', 'enemy_troop')
            if opponent_card:
                self.update_opponent_card_memory(opponent_card)

            if not self.my_last_card:
                continue

            # Track response pattern
            if self.my_last_card not in self.opponent_memory["responses_to_my_cards"]:
                self.opponent_memory["responses_to_my_cards"][self.my_last_card] = []

            self.opponent_memory["responses_to_my_cards"][self.my_last_card].append({
                "response": opponent_card,
                "reaction_time": reaction_time,
                "position": {"x": troop.get('x'), "y": troop.get('y')}
            })

            # Maintain memory size
            if len(self.opponent_memory["responses_to_my_cards"][self.my_last_card]) > 10:
                self.opponent_memory["responses_to_my_cards"][self.my_last_card].pop(0)

            # Update counter pattern statistics
            counter_key = (self.my_last_card, opponent_card)
            self.opponent_memory["counter_patterns"][counter_key] = self.opponent_memory["counter_patterns"].get(counter_key, 0) + 1

            # Track reaction time history
            self.opponent_memory["reaction_time"].append(reaction_time)
            if len(self.opponent_memory["reaction_time"]) > 30:
                self.opponent_memory["reaction_time"].pop(0)

            print(f"📊 Opponent Response: {self.my_last_card} → {opponent_card} (⏱️{reaction_time:.1f}s)")

        self.opponent_memory["last_detected_troops"] = list(opponent_troops)
    
    def get_opponent_likely_counter(self, my_card):
        """Predict what opponent will likely play to counter my card"""
        if my_card not in self.opponent_memory["responses_to_my_cards"]:
            return None
            
        responses = self.opponent_memory["responses_to_my_cards"][my_card]
        if not responses:
            return None
            
        # Get most common response
        response_counts = {}
        for response in responses:
            resp_type = response["response"]
            response_counts[resp_type] = response_counts.get(resp_type, 0) + 1
            
        most_common = max(response_counts.items(), key=lambda x: x[1])
        confidence = most_common[1] / len(responses)
        
        return {
            "counter": most_common[0],
            "confidence": confidence,
            "avg_reaction_time": sum(r["reaction_time"] for r in responses) / len(responses)
        }

    def update_opponent_card_memory(self, card_name):
        """Maintain rolling memory of opponent card sequence and usage patterns"""
        if not card_name:
            return

        card_name = card_name.lower()
        self.opponent_memory["last_cards"].append(card_name)
        if len(self.opponent_memory["last_cards"]) > 15:
            self.opponent_memory["last_cards"].pop(0)

        # Track simple frequency for predictive modeling
        self.opponent_model.setdefault("card_frequency", {})
        self.opponent_model["card_frequency"][card_name] = self.opponent_model["card_frequency"].get(card_name, 0) + 1

        # Update timing pattern estimates
        timestamp = time.time()
        self.opponent_model.setdefault("timing_patterns", []).append(timestamp)
        if len(self.opponent_model["timing_patterns"]) > 20:
            self.opponent_model["timing_patterns"].pop(0)
    
    def update_position_success(self, position, was_successful):
        """Update position success tracking"""
        pos_key = f"{position[0]:.1f},{position[1]:.1f}"
        
        if pos_key not in self.position_memory["success_heatmap"]:
            self.position_memory["success_heatmap"][pos_key] = {"successes": 0, "attempts": 0}
            
        self.position_memory["success_heatmap"][pos_key]["attempts"] += 1
        if was_successful:
            self.position_memory["success_heatmap"][pos_key]["successes"] += 1
        else:
            # Track recent failures to avoid
            self.position_memory["recent_failures"].append(pos_key)
            if len(self.position_memory["recent_failures"]) > 5:
                self.position_memory["recent_failures"].pop(0)
                
        success_rate = (self.position_memory["success_heatmap"][pos_key]["successes"] / 
                       self.position_memory["success_heatmap"][pos_key]["attempts"])
        
        print(f"📍 Position {pos_key}: {'✅' if was_successful else '❌'} (Success: {success_rate:.1%})")
    
    def get_adaptive_positions(self, card_archetype, avoid_recent_failures=True):
        """Get positions adapted to opponent behavior and success history"""
        base_positions = self.get_strategic_positions(card_archetype)
        
        if not base_positions:
            return base_positions
            
        # Filter out recently failed positions
        if avoid_recent_failures:
            filtered_positions = []
            for pos in base_positions:
                pos_key = f"{pos[0]:.1f},{pos[1]:.1f}"
                if pos_key not in self.position_memory["recent_failures"]:
                    filtered_positions.append(pos)
            if filtered_positions:  # Only use filtered if we have options
                base_positions = filtered_positions
        
        # Sort positions by success rate
        def get_success_rate(pos):
            pos_key = f"{pos[0]:.1f},{pos[1]:.1f}"
            if pos_key in self.position_memory["success_heatmap"]:
                data = self.position_memory["success_heatmap"][pos_key]
                if data["attempts"] >= 2:  # Only consider if we have enough data
                    return data["successes"] / data["attempts"]
            return 0.5  # Default neutral success rate
        
        # Add variety - don't always pick the same "best" position
        sorted_positions = sorted(base_positions, key=get_success_rate, reverse=True)
        
        # Use top 3 positions and rotate between them for variety
        top_positions = sorted_positions[:3] if len(sorted_positions) >= 3 else sorted_positions
        
        # Track position variety to avoid repetition
        for pos in top_positions:
            pos_key = f"{pos[0]:.1f},{pos[1]:.1f}"
            self.position_memory["position_variety_counter"][pos_key] = (
                self.position_memory["position_variety_counter"].get(pos_key, 0) + 1
            )
        
        # Prefer less recently used positions for variety
        variety_sorted = sorted(top_positions, 
                              key=lambda p: self.position_memory["position_variety_counter"].get(
                                  f"{p[0]:.1f},{p[1]:.1f}", 0))
        
        return variety_sorted
    
    def analyze_meta_game_state(self, current_elixir, enemy_troops, our_troops):
        """Analyze the current meta-game state for strategic decisions"""
        current_time = time.time()
        game_duration = current_time - self.game_state["game_start_time"]
        
        # 🕒 UPDATE GAME PHASE
        if game_duration < 60:  # 0-1 minute
            self.game_state["game_phase"] = "EARLY"
        elif game_duration < 120:  # 1-2 minutes
            self.game_state["game_phase"] = "MID"
        elif game_duration < 180:  # 2-3 minutes
            self.game_state["game_phase"] = "LATE"
        else:
            self.game_state["game_phase"] = "OVERTIME"
        
        # ⚡ ESTIMATE ELIXIR ADVANTAGE
        # Simple estimation based on recent plays and current elixir
        time_since_last_play = current_time - self.my_last_play_time
        elixir_generation = min(time_since_last_play * 1.4, 4)  # ~1.4 elixir per second, max 10
        estimated_opponent_elixir = min(5 + elixir_generation, 10)  # Rough estimate
        self.game_state["elixir_advantage"] = current_elixir - estimated_opponent_elixir
        
        # 📊 ANALYZE MOMENTUM
        enemy_threat_level = len(enemy_troops) + sum(1 for troop in enemy_troops if troop.get('y', 0) > 0.6)
        our_pressure_level = len(our_troops) + sum(1 for troop in our_troops if troop.get('y', 0) < 0.4)
        
        if enemy_threat_level > our_pressure_level + 2:
            new_momentum = "LOSING"
        elif our_pressure_level > enemy_threat_level + 2:
            new_momentum = "WINNING"
        else:
            new_momentum = "NEUTRAL"
            
        if new_momentum != self.game_state["momentum"]:
            self.game_state["momentum"] = new_momentum
            self.game_state["last_momentum_change"] = current_time
            print(f"🔄 Momentum shift: {new_momentum} (Threats: {enemy_threat_level}, Pressure: {our_pressure_level})")
        
        # 🎯 PRESSURE LEVEL ANALYSIS
        if enemy_threat_level >= 3:
            self.game_state["pressure_level"] = "HIGH"
        elif enemy_threat_level >= 2:
            self.game_state["pressure_level"] = "MEDIUM"
        elif enemy_threat_level >= 1:
            self.game_state["pressure_level"] = "LOW"
        else:
            self.game_state["pressure_level"] = "NONE"
    
    def determine_action_context(self, strategic_decision, enemy_troops):
        """Determine the best action context based on game state"""
        current_time = time.time()
        
        # 🛡️ FORCED DEFENSE CHECK
        immediate_threats = [t for t in enemy_troops if t.get('y', 0) > 0.7]
        self.action_context["forced_defense"] = len(immediate_threats) > 0
        
        if self.action_context["forced_defense"]:
            self.action_context["mode"] = "REACTIVE"
            print(f"⚠️ Forced defense mode: {len(immediate_threats)} immediate threats")
            return
        
        # 🔄 MODE SELECTION BASED ON GAME STATE
        if self.game_state["momentum"] == "LOSING":
            if self.game_state["elixir_advantage"] > 2:
                self.action_context["mode"] = "AGGRESSIVE"  # We have elixir, push back
            else:
                self.action_context["mode"] = "DEFENSIVE"  # Survive and stabilize
                
        elif self.game_state["momentum"] == "WINNING":
            if self.game_state["pressure_level"] == "NONE":
                self.action_context["mode"] = "PROACTIVE"  # Apply pressure
            else:
                self.action_context["mode"] = "BALANCED"  # Maintain advantage
                
        else:  # NEUTRAL momentum
            if self.game_state["game_phase"] == "EARLY":
                self.action_context["mode"] = "BALANCED"  # Feel out opponent
            elif self.game_state["game_phase"] in ["LATE", "OVERTIME"]:
                if self.game_state["tower_advantage"] > 0:
                    self.action_context["mode"] = "DEFENSIVE"  # Protect lead
                else:
                    self.action_context["mode"] = "AGGRESSIVE"  # Need to win
            else:
                self.action_context["mode"] = "PROACTIVE"  # Mid-game pressure
        
        # 🎯 SPECIAL CONTEXTS
        time_since_opponent_aggression = current_time - self.action_context["last_opponent_aggression"]
        if len(enemy_troops) == 0 and time_since_opponent_aggression > 5:
            self.action_context["counter_window"] = True  # Safe to attack
        else:
            self.action_context["counter_window"] = False
            
        # Track opponent aggression
        if len(enemy_troops) > 0:
            self.action_context["last_opponent_aggression"] = current_time
        
        print(f"🎮 Action Context: {self.action_context['mode']} | Phase: {self.game_state['game_phase']} | Momentum: {self.game_state['momentum']}")

    def select_high_level_strategy(self, strategic_decision):
        """🏗️ Hierarchical strategy selection guiding low-level actions"""
        available_strategies = ["PRESSURE", "CONTROL", "RUSH", "STALL", "BALANCED"]
        chosen_strategy = strategic_decision

        # Meta-based overrides
        detected_meta = self.meta_adaptation.get("detected_meta", "UNKNOWN")
        if detected_meta == "BEATDOWN":
            chosen_strategy = "PRESSURE"
        elif detected_meta == "CONTROL":
            chosen_strategy = "RUSH"
        elif detected_meta == "CYCLE":
            chosen_strategy = "CONTROL"

        # Explore alternative strategies occasionally
        if random.random() < self.strategy_hierarchy["explore_probability"]:
            chosen_strategy = random.choice(available_strategies)

        self.strategy_hierarchy["high_level_strategy"] = chosen_strategy
        self.strategy_hierarchy["recent_strategies"].append(chosen_strategy)
        if len(self.strategy_hierarchy["recent_strategies"]) > 15:
            self.strategy_hierarchy["recent_strategies"].pop(0)

        print(f"🏗️ High-level strategy: {chosen_strategy}")
        return chosen_strategy

    def apply_real_time_adaptations(self, opponent_troops):
        """⚡ Apply real-time adaptation logic based on opponent behavior"""
        if not opponent_troops:
            return

        likely_counter = None
        if self.my_last_card:
            counter_info = self.get_opponent_likely_counter(self.my_last_card)
            if counter_info and counter_info.get("confidence", 0) > 0.5:
                likely_counter = counter_info.get("counter")

        if likely_counter:
            self.strategy_patterns["deception_mode"] = True
            self.deception["next_bait_target"] = self.my_last_card
            print(f"⚡ Anticipating counter '{likely_counter}', preparing deception")

        # Lane exploitation
        lane_pref = self.opponent_memory.get("preferred_lanes", {})
        if lane_pref:
            weakest_lane = min(lane_pref, key=lane_pref.get)
            self.position_memory["opponent_weakness_map"][weakest_lane] = min(
                self.position_memory["opponent_weakness_map"].get(weakest_lane, 0.5) + 0.1,
                1.0
            )

        # Update strategic mixing payoffs (simplified reward estimation)
        self.update_strategy_mixing()
    
    def get_contextual_card_choice(self, available_cards, strategic_decision):
        """🎯 Master card selection with all advanced AI systems"""
        if not available_cards:
            return None
            
        playable_cards = self.get_playable_cards(available_cards)
        if not playable_cards:
            return None
        
        # 🎲 RANDOMIZED STRATEGY ANALYSIS
        self.analyze_randomized_strategy()
        
        # 🔮 PREDICTIVE OPPONENT MODELING
        enemy_troops = []  # Would be populated from game state
        self.predict_opponent_behavior(available_cards, enemy_troops)
        
        # 🎭 DECEPTION STRATEGY CHECK
        deception_card = self.execute_deception_strategy(playable_cards)
        if deception_card and self.deception["misdirection_active"]:
            # Evaluate risk of deception play
            risk_assessment = self.assess_risk_reward(deception_card, (0.5, 0.5), strategic_decision)
            if risk_assessment["recommendation"] in ["EXECUTE", "CONSIDER"]:
                print(f"🎭 Deception override: {deception_card}")
                return deception_card
        
        # 🔄 COUNTER-META ANALYSIS
        counter_meta_card = self.detect_and_counter_meta(enemy_troops, playable_cards)
        if counter_meta_card:
            risk_assessment = self.assess_risk_reward(counter_meta_card, (0.5, 0.5), strategic_decision)
            if risk_assessment["recommendation"] == "EXECUTE":
                print(f"🔄 Counter-meta override: {counter_meta_card}")
                return counter_meta_card
        
        # 🧩 COMBO OPPORTUNITY ANALYSIS
        combo_card = self.analyze_combo_opportunities(playable_cards)
        if combo_card:
            risk_assessment = self.assess_risk_reward(combo_card, (0.5, 0.5), strategic_decision)
            if risk_assessment["recommendation"] in ["EXECUTE", "CONSIDER"]:
                print(f"🧩 Combo override: {combo_card}")
                return combo_card
        
        # Get base adaptive choice
        base_choice = self.get_adaptive_card_choice(available_cards, strategic_decision)
        if not base_choice:
            base_choice = playable_cards[0] if playable_cards else None

        if base_choice:
            predicted_counter = self.get_opponent_likely_counter(base_choice)
            if predicted_counter and predicted_counter.get("confidence", 0) > 0.6:
                counter_play = self.get_counter_to_counter(predicted_counter.get("counter"), playable_cards)
                if counter_play and counter_play != base_choice:
                    print(
                        f"🛡️ Counter-counter adjustment: {base_choice} → {counter_play} "
                        f"(anticipating {predicted_counter.get('counter')})"
                    )
                    base_choice = counter_play
            
        # 🎯 CONTEXTUAL CARD SELECTION
        context_mode = self.action_context["mode"]
        high_level = self.strategy_hierarchy.get("high_level_strategy", "BALANCED")

        if high_level == "RUSH" and playable_cards:
            heavy_hitters = [c for c in playable_cards if self.get_card_cost(c) >= 5]
            if heavy_hitters and self.current_elixir >= self.get_card_cost(heavy_hitters[0]):
                print(f"🏃 Rush strategy prioritizing {heavy_hitters[0]}")
                return heavy_hitters[0]
        elif high_level == "CONTROL" and playable_cards:
            control_cards = [c for c in playable_cards if self.detect_card_archetype(c) in ["SPELLS", "BUILDINGS"]]
            if control_cards:
                print(f"🛡️ Control strategy selecting {control_cards[0]}")
                return control_cards[0]
        elif high_level == "STALL" and playable_cards:
            cheap_cycle = [c for c in playable_cards if self.get_card_cost(c) <= 3]
            if cheap_cycle:
                print(f"⏳ Stall strategy cycling {cheap_cycle[0]}")
                return cheap_cycle[0]
        
        if context_mode == "REACTIVE":
            # Prioritize cheap defensive cards
            defensive_cards = [c for c in playable_cards 
                             if self.detect_card_archetype(c) in ["SWARM", "BUILDINGS", "ANTI_AIR"]
                             and self.get_card_cost(c) <= 4]
            if defensive_cards:
                choice = min(defensive_cards, key=self.get_card_cost)
                print(f"🛡️ Reactive choice: {choice} (defensive)")
                return choice
                
        elif context_mode == "AGGRESSIVE":
            # Prioritize win conditions and damage dealers
            aggressive_cards = [c for c in playable_cards 
                              if self.detect_card_archetype(c) in ["WIN_CONDITIONS", "TANKS", "SPELLS"]
                              and self.get_card_cost(c) <= self.current_elixir]
            if aggressive_cards and self.current_elixir >= 5:
                choice = aggressive_cards[0]
                print(f"⚔️ Aggressive choice: {choice} (offensive)")
                return choice
                
        elif context_mode == "PROACTIVE":
            # Mix of pressure and cycle cards
            if self.action_context["counter_window"]:
                # Safe to play big cards
                big_cards = [c for c in playable_cards if self.get_card_cost(c) >= 4]
                if big_cards:
                    choice = big_cards[0]
                    print(f"🎯 Proactive choice: {choice} (counter-window)")
                    return choice
            else:
                # Play cheaper cards for cycle
                cheap_cards = [c for c in playable_cards if self.get_card_cost(c) <= 3]
                if cheap_cards:
                    choice = cheap_cards[0]
                    print(f"🔄 Proactive choice: {choice} (cycle)")
                    return choice
                    
        elif context_mode == "DEFENSIVE":
            # Prioritize elixir efficiency and defense
            efficient_cards = [c for c in playable_cards 
                             if self.get_card_cost(c) <= 4]
            if efficient_cards:
                choice = min(efficient_cards, key=self.get_card_cost)
                print(f"🏰 Defensive choice: {choice} (efficient)")
                return choice
        
        # ⚖️ FINAL RISK-REWARD ASSESSMENT
        if base_choice:
            final_risk_assessment = self.assess_risk_reward(base_choice, (0.5, 0.5), strategic_decision)
            
            if final_risk_assessment["recommendation"] == "AVOID":
                # Find safer alternative
                safe_alternatives = []
                for card in playable_cards:
                    if card != base_choice:
                        alt_risk = self.assess_risk_reward(card, (0.5, 0.5), strategic_decision)
                        if alt_risk["recommendation"] in ["EXECUTE", "CONSIDER", "NEUTRAL"]:
                            safe_alternatives.append((card, alt_risk["ratio"]))
                
                if safe_alternatives:
                    # Choose best safe alternative
                    safe_alternatives.sort(key=lambda x: x[1], reverse=True)
                    safer_choice = safe_alternatives[0][0]
                    print(f"⚖️ Risk-adjusted choice: {base_choice} → {safer_choice} (safer)")
                    return safer_choice
            
            print(f"🎯 Final contextual choice: {base_choice} (risk: {final_risk_assessment['risk']:.2f}, reward: {final_risk_assessment['reward']:.2f})")
        
        return base_choice

    def update_strategy_mixing(self):
        """🚀 Update mixed strategy distribution using regret-style adjustments"""
        payoffs = {
            "ATTACK": self.objectives.get("successful_attacks", 0),
            "DEFEND": self.objectives.get("successful_defenses", 0),
            "PRESSURE": len([c for c in self.cards_played_this_game[-5:] if c])
        }

        avg_payoff = sum(payoffs.values()) / max(len(payoffs), 1)
        for strategy, payoff in payoffs.items():
            regret = payoff - avg_payoff
            self.strategy_mixing["regret_minimization"][strategy] = (
                self.strategy_mixing["regret_minimization"].get(strategy, 0.0) + regret
            )

        positive_regrets = {
            k: max(v, 0.0) for k, v in self.strategy_mixing["regret_minimization"].items()
        }
        total_regret = sum(positive_regrets.values())
        if total_regret > 0:
            for strategy in self.strategy_mixing["mixed_strategy"].keys():
                self.strategy_mixing["mixed_strategy"][strategy] = positive_regrets.get(strategy, 0.0) / total_regret
        else:
            # Default back to uniform mixing
            for strategy in self.strategy_mixing["mixed_strategy"].keys():
                self.strategy_mixing["mixed_strategy"][strategy] = 1 / len(self.strategy_mixing["mixed_strategy"])

        self.strategy_mixing["last_payoffs"].append(payoffs)
        if len(self.strategy_mixing["last_payoffs"]) > self.strategy_mixing.get("history_window", 10):
            self.strategy_mixing["last_payoffs"].pop(0)
    
    def update_multi_objectives(self, prev_state, current_state, card_played, reward):
        """Update multiple learning objectives"""
        if prev_state is None or current_state is None:
            return
            
        # 💥 DAMAGE TRACKING
        prev_enemy_presence = sum(prev_state[1 + 2 * MAX_ALLIES:][1::2])
        current_enemy_presence = sum(current_state[1 + 2 * MAX_ALLIES:][1::2])
        enemy_reduction = prev_enemy_presence - current_enemy_presence
        
        if enemy_reduction > 0:
            self.objectives["damage_dealt"] += enemy_reduction
            self.objectives["successful_attacks"] += 1
            
        # 🛡️ DEFENSE TRACKING
        prev_enemy_threats = sum(1 for i in range(1 + 2 * MAX_ALLIES, len(prev_state), 2)
                               if i + 1 < len(prev_state) and prev_state[i + 1] > 0.6)
        current_enemy_threats = sum(1 for i in range(1 + 2 * MAX_ALLIES, len(current_state), 2)
                                  if i + 1 < len(current_state) and current_state[i + 1] > 0.6)
        
        if prev_enemy_threats > current_enemy_threats:
            self.objectives["successful_defenses"] += 1
            
        # ⚡ ELIXIR EFFICIENCY
        if card_played:
            card_cost = self.get_card_cost(card_played)
            efficiency = enemy_reduction / max(card_cost, 1)  # Damage per elixir
            self.objectives["elixir_efficiency"].append(efficiency)
            
            # Keep only last 10 trades
            if len(self.objectives["elixir_efficiency"]) > 10:
                self.objectives["elixir_efficiency"].pop(0)
                
        # 🎯 ADAPTATION SCORE
        if hasattr(self, 'my_last_card') and self.my_last_card:
            counter_info = self.get_opponent_likely_counter(self.my_last_card)
            if counter_info and counter_info["confidence"] > 0.5:
                # We're adapting if we avoid heavily countered cards
                avoided_counter = card_played != self.my_last_card
                if avoided_counter:
                    self.objectives["adaptation_score"] += 1
                    
        # 📍 POSITION VARIETY SCORE
        if len(self.positions_used_this_game) > 1:
            recent_positions = self.positions_used_this_game[-5:]  # Last 5 positions
            unique_positions = len(set((round(p[0], 1), round(p[1], 1)) for p in recent_positions))
            variety_ratio = unique_positions / len(recent_positions)
            self.objectives["position_variety_score"] = variety_ratio
            
        # Print objectives summary every 5 plays
        if len(self.cards_played_this_game) % 5 == 0:
            avg_efficiency = sum(self.objectives["elixir_efficiency"]) / max(len(self.objectives["elixir_efficiency"]), 1)
            print(f"📊 Multi-Objectives: Attacks: {self.objectives['successful_attacks']}, "
                  f"Defenses: {self.objectives['successful_defenses']}, "
                  f"Efficiency: {avg_efficiency:.2f}, "
                  f"Adaptation: {self.objectives['adaptation_score']}, "
                  f"Variety: {self.objectives['position_variety_score']:.2f}")

        if reward > 0 and self.cards_played_this_game:
            recent_sequence = tuple(self.cards_played_this_game[-min(4, len(self.cards_played_this_game)):])
            if recent_sequence:
                self.imitation_memory["successful_sequences"].append(recent_sequence)
                if len(self.imitation_memory["successful_sequences"]) > 25:
                    self.imitation_memory["successful_sequences"].pop(0)

        current_strategy = self.strategy_hierarchy.get("high_level_strategy")
        if current_strategy:
            self.strategy_hierarchy["strategy_success"][current_strategy] = (
                self.strategy_hierarchy["strategy_success"].get(current_strategy, 0) * 0.9 + max(reward, 0)
            )

        self.strategy_patterns["pattern_success"].append(1 if reward > 0 else 0)
        if len(self.strategy_patterns["pattern_success"]) > 20:
            self.strategy_patterns["pattern_success"].pop(0)
    
    def initialize_combo_database(self):
        """Initialize known card combinations and their synergies"""
        self.combo_system["available_combos"] = {
            "giant_push": ["giant", "wizard", "minions"],
            "hog_cycle": ["hog rider", "ice spirit", "skeletons", "ice golem"],
            "balloon_rage": ["balloon", "rage", "freeze"],
            "golem_beatdown": ["golem", "night witch", "baby dragon", "lightning"],
            "royal_giant_support": ["royal giant", "furnace", "minions", "zap"],
            "miner_poison": ["miner", "poison", "inferno tower"],
            "lava_loon": ["lavahound", "balloon", "minions", "tombstone"],
            "bridge_spam": ["battle ram", "bandit", "ghost", "poison"],
            "spell_bait": ["goblin barrel", "princess", "goblin gang", "inferno tower"],
            "graveyard_poison": ["graveyard", "poison", "knight", "archers"]
        }
        
        # Initialize success rates
        for combo_name in self.combo_system["available_combos"]:
            self.combo_system["combo_success_rate"][combo_name] = 0.5  # Start neutral
            self.combo_system["combo_readiness"][combo_name] = 0.0
            self.combo_system["combo_counter_ready"][combo_name] = False
    
    def analyze_randomized_strategy(self):
        """🎲 Implement randomized strategy patterns for unpredictability"""
        current_time = time.time()
        pattern_duration = current_time - self.strategy_patterns["last_pattern_change"]
        
        # Change pattern every 30-60 seconds or if current pattern is failing
        should_change_pattern = (
            pattern_duration > random.uniform(30, 60) or
            (len(self.strategy_patterns["pattern_success"]) > 3 and 
             sum(self.strategy_patterns["pattern_success"][-3:]) < 1)  # Last 3 attempts failed
        )
        
        if should_change_pattern:
            # Choose new pattern based on game state and randomization
            patterns = ["AGGRESSIVE", "DEFENSIVE", "CYCLE", "ADAPTIVE"]
            
            if random.random() < self.strategy_patterns["randomization_level"]:
                # Random choice for unpredictability
                new_pattern = random.choice(patterns)
                print(f"🎲 Random strategy switch: {self.strategy_patterns['current_pattern']} → {new_pattern}")
            else:
                # Strategic choice based on game state
                if self.game_state["momentum"] == "LOSING":
                    new_pattern = "AGGRESSIVE" if self.game_state["elixir_advantage"] > 0 else "DEFENSIVE"
                elif self.game_state["momentum"] == "WINNING":
                    new_pattern = "CYCLE" if self.game_state["game_phase"] == "LATE" else "ADAPTIVE"
                else:
                    new_pattern = "ADAPTIVE"
                print(f"🧠 Strategic pattern switch: {self.strategy_patterns['current_pattern']} → {new_pattern}")
            
            self.strategy_patterns["current_pattern"] = new_pattern
            self.strategy_patterns["last_pattern_change"] = current_time
            self.strategy_patterns["pattern_duration"] = 0
        
        # Activate deception mode occasionally
        if random.random() < 0.1 and not self.strategy_patterns["deception_mode"]:
            self.strategy_patterns["deception_mode"] = True
            self.deception["misdirection_active"] = True
            print(f"🎭 Activating deception mode for unpredictability")
    
    def predict_opponent_behavior(self, current_cards, enemy_troops):
        """🔮 Predictive opponent modeling system"""
        current_time = time.time()
        
        # Track opponent play patterns
        if len(enemy_troops) > 0:
            enemy_types = [troop.get('class', 'unknown') for troop in enemy_troops]
            pattern_key = ",".join(sorted(enemy_types))

            self.opponent_model.setdefault("play_patterns", {})
            self.opponent_model["play_patterns"][pattern_key] = self.opponent_model["play_patterns"].get(pattern_key, 0) + 1

        # Predict next likely play using combined evidence
        predicted_card = None
        confidence = 0.0

        if self.opponent_model.get("card_frequency"):
            freq_map = self.opponent_model["card_frequency"]
            predicted_card = max(freq_map, key=freq_map.get)
            total_freq = sum(freq_map.values()) or 1
            confidence = freq_map[predicted_card] / total_freq

        if self.opponent_model.get("play_patterns"):
            pattern_counts = self.opponent_model["play_patterns"]
            most_common_pattern = max(pattern_counts, key=pattern_counts.get)
            pattern_total = sum(pattern_counts.values()) or 1
            pattern_confidence = pattern_counts[most_common_pattern] / pattern_total
            if pattern_confidence > confidence:
                split_cards = most_common_pattern.split(",")
                predicted_card = split_cards[0] if split_cards else predicted_card
                confidence = pattern_confidence

        self.opponent_model["prediction_confidence"] = confidence
        self.opponent_model["predicted_next_card"] = predicted_card

        if predicted_card and confidence > 0.5:
            print(f"🔮 Predicting opponent will play: {predicted_card} (confidence: {confidence:.2f})")
        
        # Predict aggression level
        recent_enemy_count = len(enemy_troops)
        if recent_enemy_count >= 3:
            self.opponent_model["aggression_prediction"] = "AGGRESSIVE"
        elif recent_enemy_count <= 1:
            self.opponent_model["aggression_prediction"] = "DEFENSIVE"
        else:
            self.opponent_model["aggression_prediction"] = "NEUTRAL"
    
    def execute_deception_strategy(self, available_cards):
        """🎭 Implement deception and bluffing tactics"""
        if not self.deception["misdirection_active"]:
            return None
            
        # Bait strategy: Play weak card to bait opponent's counter
        if self.deception["next_bait_target"] is None and available_cards:
            # Choose a bait target (common counter cards)
            bait_targets = ["zap", "arrows", "fireball", "lightning", "rocket"]
            potential_baits = [card for card in available_cards 
                             if any(target in self.get_card_counters(card) for target in bait_targets)]
            
            if potential_baits:
                self.deception["next_bait_target"] = random.choice(potential_baits)
                print(f"🎭 Setting up bait with: {self.deception['next_bait_target']}")
        
        # Hide our strongest cards early game
        if (self.game_state["game_phase"] == "EARLY" and 
            self.deception["hidden_strength"] is None):
            
            strong_cards = [card for card in available_cards if self.get_card_cost(card) >= 5]
            if strong_cards:
                self.deception["hidden_strength"] = random.choice(strong_cards)
                print(f"🎭 Hiding strength card: {self.deception['hidden_strength']}")
        
        # Execute bait if conditions are right
        if (self.deception["next_bait_target"] and 
            self.deception["next_bait_target"] in available_cards and
            self.current_elixir >= self.get_card_cost(self.deception["next_bait_target"])):
            
            bait_card = self.deception["next_bait_target"]
            self.deception["bait_attempts"] += 1
            self.deception["next_bait_target"] = None
            print(f"🎣 Executing bait with: {bait_card}")
            return bait_card
            
        return None
    
    def assess_risk_reward(self, card, position, strategic_decision):
        """⚖️ Advanced risk-reward assessment system"""
        if not card:
            return {"risk": 1.0, "reward": 0.0, "recommendation": "AVOID"}
            
        card_cost = self.get_card_cost(card)
        
        # Base risk factors
        elixir_risk = min(card_cost / self.current_elixir, 1.0) if self.current_elixir > 0 else 1.0
        position_risk = 0.3 if position[1] < 0.5 else 0.1  # Aggressive positioning is riskier
        
        # Contextual risk factors
        momentum_risk = {
            "LOSING": 0.8,  # High risk when losing
            "NEUTRAL": 0.5,
            "WINNING": 0.2   # Low risk when winning
        }.get(self.game_state["momentum"], 0.5)
        
        phase_risk = {
            "EARLY": 0.3,    # Low risk early game
            "MID": 0.5,
            "LATE": 0.7,     # Higher risk late game
            "OVERTIME": 0.9  # Very high risk in overtime
        }.get(self.game_state["game_phase"], 0.5)
        
        # Calculate composite risk
        total_risk = (elixir_risk * 0.3 + position_risk * 0.2 + 
                     momentum_risk * 0.3 + phase_risk * 0.2)
        
        # Reward factors
        strategic_reward = {
            "ATTACK": 0.8,
            "DEFEND": 0.6,
            "PRESSURE": 0.7,
            "PREVENT_LEAK": 0.9,
            "WAIT": 0.2
        }.get(strategic_decision, 0.5)
        
        # Position success bonus
        pos_key = f"{position[0]:.1f},{position[1]:.1f}"
        if pos_key in self.position_memory:
            position_reward = self.position_memory[pos_key]["success_rate"]
        else:
            position_reward = 0.5
            
        total_reward = strategic_reward * 0.6 + position_reward * 0.4
        
        # Risk tolerance adjustment
        risk_tolerance = self.risk_assessment["risk_tolerance"]
        if self.game_state["momentum"] == "LOSING":
            risk_tolerance *= 1.5  # More willing to take risks when losing
        
        # Recommendation
        risk_reward_ratio = total_reward / max(total_risk, 0.1)
        
        if risk_reward_ratio > 1.5 and total_risk <= risk_tolerance:
            recommendation = "EXECUTE"
        elif risk_reward_ratio > 1.0 and total_risk <= risk_tolerance * 0.8:
            recommendation = "CONSIDER"
        elif total_risk > risk_tolerance * 1.2:
            recommendation = "AVOID"
        else:
            recommendation = "NEUTRAL"
            
        self.risk_assessment["recent_risks"].append({
            "card": card,
            "risk": total_risk,
            "reward": total_reward,
            "ratio": risk_reward_ratio
        })
        
        # Keep only last 10 risk assessments
        if len(self.risk_assessment["recent_risks"]) > 10:
            self.risk_assessment["recent_risks"].pop(0)
            
        return {
            "risk": total_risk,
            "reward": total_reward,
            "ratio": risk_reward_ratio,
            "recommendation": recommendation
        }
    
    def detect_and_counter_meta(self, enemy_troops, available_cards):
        """🔄 Counter-meta adaptation system"""
        if not enemy_troops:
            return None
            
        enemy_types = [troop.get('class', 'unknown') for troop in enemy_troops]
        
        # Meta detection patterns
        meta_signatures = {
            "BEATDOWN": ["golem", "giant", "lavahound", "royal giant"],
            "CYCLE": ["hog rider", "ice spirit", "skeletons", "ice golem"],
            "CONTROL": ["xbow", "mortar", "inferno tower", "tesla"],
            "SIEGE": ["xbow", "mortar", "rocket", "log"],
            "BRIDGE_SPAM": ["battle ram", "bandit", "ghost", "dark prince"]
        }
        
        # Check for meta signatures
        for meta_type, signature_cards in meta_signatures.items():
            matches = sum(1 for enemy in enemy_types if any(sig in enemy.lower() for sig in signature_cards))
            match_ratio = matches / len(signature_cards)
            
            if match_ratio > 0.4:  # 40% signature match
                if self.meta_adaptation["detected_meta"] != meta_type:
                    self.meta_adaptation["detected_meta"] = meta_type
                    self.meta_adaptation["meta_confidence"] = match_ratio
                    print(f"🔄 Detected opponent meta: {meta_type} (confidence: {match_ratio:.2f})")
                    
                    # Set counter strategy
                    counter_strategies = {
                        "BEATDOWN": "PRESSURE",  # Pressure opposite lane
                        "CYCLE": "CONTROL",     # Control their cycle
                        "CONTROL": "BEATDOWN",  # Overwhelming force
                        "SIEGE": "RUSH",       # Rush their defenses
                        "BRIDGE_SPAM": "DEFEND" # Strong defense first
                    }
                    
                    self.meta_adaptation["counter_strategy"] = counter_strategies.get(meta_type, "ADAPTIVE")
                    print(f"🎯 Counter-strategy: {self.meta_adaptation['counter_strategy']}")
                    
                    # Select counter card if available
                    counter_cards = {
                        "BEATDOWN": ["inferno tower", "inferno dragon", "minion horde"],
                        "CYCLE": ["tornado", "ice wizard", "bowler"],
                        "CONTROL": ["lightning", "earthquake", "royal ghost"],
                        "SIEGE": ["rocket", "lightning", "miner"],
                        "BRIDGE_SPAM": ["valkyrie", "mega knight", "log"]
                    }
                    
                    available_counters = [card for card in available_cards 
                                        if any(counter in card.lower() 
                                             for counter in counter_cards.get(meta_type, []))]
                    
                    if available_counters:
                        counter_card = available_counters[0]
                        print(f"🛡️ Counter-meta card selected: {counter_card}")
                        return counter_card
                        
        return None
    
    def analyze_combo_opportunities(self, available_cards):
        """🧩 Advanced combo chain recognition and execution"""
        if not available_cards:
            return None
            
        best_combo = None
        best_combo_score = 0
        
        for combo_name, combo_cards in self.combo_system["available_combos"].items():
            # Check how many combo cards we have available
            available_combo_cards = [card for card in available_cards 
                                   if any(combo_card.lower() in card.lower() 
                                        for combo_card in combo_cards)]
            
            if len(available_combo_cards) >= 2:  # Need at least 2 cards for combo
                # Calculate combo readiness
                total_elixir_needed = sum(self.get_card_cost(card) for card in available_combo_cards)
                elixir_readiness = min(self.current_elixir / total_elixir_needed, 1.0) if total_elixir_needed > 0 else 0
                
                # Factor in combo success rate
                success_rate = self.combo_system["combo_success_rate"].get(combo_name, 0.5)
                
                # Calculate combo score
                combo_score = (len(available_combo_cards) / len(combo_cards)) * elixir_readiness * success_rate
                
                if combo_score > best_combo_score and combo_score > 0.6:  # Minimum threshold
                    best_combo = combo_name
                    best_combo_score = combo_score
                    self.combo_system["current_combo_setup"] = combo_name
        
        if best_combo:
            combo_cards = self.combo_system["available_combos"][best_combo]
            available_combo_cards = [card for card in available_cards 
                                   if any(combo_card.lower() in card.lower() 
                                        for combo_card in combo_cards)]
            
            if available_combo_cards:
                first_card = available_combo_cards[0]
                print(f"🧩 Executing combo '{best_combo}' starting with: {first_card} (score: {best_combo_score:.2f})")
                return first_card
                
        return None
    
    def get_card_counters(self, card):
        """Get list of cards that commonly counter the given card"""
        counter_database = {
            "goblin barrel": ["zap", "arrows", "log"],
            "minion horde": ["arrows", "zap", "fireball"],
            "skeleton army": ["zap", "arrows", "log"],
            "inferno tower": ["zap", "lightning", "earthquake"],
            "xbow": ["rocket", "lightning", "earthquake"],
            "golem": ["inferno tower", "inferno dragon"],
            "giant": ["inferno tower", "mini pekka"],
            "balloon": ["musketeer", "archers", "inferno tower"],
            "hog rider": ["cannon", "tesla", "tombstone"],
            "royal giant": ["inferno tower", "barbarians"]
        }
        
        for card_name, counters in counter_database.items():
            if card_name.lower() in card.lower():
                return counters
        return []
    
    def get_counter_to_counter(self, predicted_counter, available_cards):
        """Determine which of our cards can counter the opponent's expected counter"""
        if not predicted_counter or not available_cards:
            return None

        counter_lower = predicted_counter.lower()
        for key, answers in self.counter_counter_map.items():
            if key in counter_lower:
                for answer in answers:
                    for card in available_cards:
                        if answer in card.lower():
                            return card
        return None

    def get_adaptive_card_choice(self, available_cards, strategy):
        """Choose card considering opponent patterns and adaptation"""
        base_choice = self.get_best_elixir_card(available_cards, strategy)
        if not base_choice:
            return None
            
        # 🧠 OPPONENT ADAPTATION LOGIC
        
        # 1. Avoid cards that opponent consistently counters well
        risky_cards = []
        for card in available_cards:
            counter_info = self.get_opponent_likely_counter(card)
            if counter_info and counter_info["confidence"] > 0.7:  # High confidence counter
                risky_cards.append(card)
                print(f"⚠️ {card} likely countered by {counter_info['counter']} ({counter_info['confidence']:.1%})")
        
        # 2. Prefer cards that have been working well
        successful_cards = []
        for card in available_cards:
            if card in self.cards_played_this_game:
                # Check if this card type has been successful recently
                # (simplified: assume success if we've used it multiple times)
                usage_count = self.cards_played_this_game.count(card)
                if usage_count >= 2:  # We keep using it, so it's probably working
                    successful_cards.append(card)
        
        # 3. Card Selection Priority:
        # - If strategy is urgent (PREVENT_LEAK, DEFEND), use base choice
        if strategy in ["PREVENT_LEAK", "DEFEND"]:
            return base_choice
            
        # - Otherwise, try to be adaptive
        safe_cards = [c for c in available_cards if c not in risky_cards]
        
        if successful_cards and strategy in ["COUNTER_PUSH", "PRESSURE"]:
            # Use cards that have been working
            preferred = [c for c in successful_cards if c in safe_cards]
            if preferred:
                choice = preferred[0]
                print(f"🎯 Adaptive choice: {choice} (proven successful)")
                return choice
                
        if safe_cards and len(safe_cards) > 1:
            # Use safe cards that aren't heavily countered
            choice = safe_cards[0] if base_choice in risky_cards else base_choice
            if choice != base_choice:
                print(f"🛡️ Adaptive choice: {choice} (avoiding counter)")
            return choice
            
        return base_choice
    
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

        tank_count = 0
        swarm_count = 0
        control_structures = 0
        lane_counts = {"left": 0, "right": 0, "center": 0}

        for troop in enemy_troops:
            x = troop.get('x', 0)
            y = troop.get('y', 0)
            troop_class = troop.get('class', '').lower()

            # Lane analysis based on x coordinate
            if x < self.actions.WIDTH * 0.33:
                lane_counts["left"] += 1
            elif x > self.actions.WIDTH * 0.66:
                lane_counts["right"] += 1
            else:
                lane_counts["center"] += 1

            # Archetype heuristics
            if any(keyword in troop_class for keyword in ["giant", "golem", "hound", "pekka", "tank"]):
                tank_count += 1
            if any(keyword in troop_class for keyword in ["minion", "skeleton", "goblin", "recruits", "swarm"]):
                swarm_count += 1
            if any(keyword in troop_class for keyword in ["mortar", "xbow", "tesla", "inferno"]):
                control_structures += 1

        # Update lane preferences memory
        for lane, count in lane_counts.items():
            self.opponent_memory["preferred_lanes"][lane] = self.opponent_memory["preferred_lanes"].get(lane, 0) + count

        # Determine archetype with simple majority
        if tank_count >= max(swarm_count, control_structures) and tank_count > 0:
            self.opponent_archetype = "BEATDOWN"
        elif control_structures > max(tank_count, swarm_count):
            self.opponent_archetype = "CONTROL"
        elif swarm_count > 1:
            self.opponent_archetype = "SWARM"
        else:
            self.opponent_archetype = "BALANCED"

        # Update opponent profile history for meta-learning
        self.opponent_profiles["profile_history"].append(self.opponent_archetype)
        if len(self.opponent_profiles["profile_history"]) > 20:
            self.opponent_profiles["profile_history"].pop(0)

        if self.opponent_profiles["profile_history"]:
            history = self.opponent_profiles["profile_history"]
            most_common_profile = max(set(history), key=history.count)
            confidence = history.count(most_common_profile) / len(history)
            self.opponent_profiles["current_profile"] = most_common_profile
            self.opponent_profiles["profile_confidence"] = confidence
    
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
        """Make strategic decision based on game state and elixir management"""
        # Update elixir management first
        self.update_elixir_management()
        
        # Analyze opponent troops
        self.analyze_opponent_troops(enemy_troops)
        
        # Enhanced Priority System with Elixir Awareness
        
        # Priority 1: Critical elixir leak prevention (10 elixir)
        if self.elixir_urgency == "CRITICAL":
            return "PREVENT_LEAK"
        
        # Priority 2: Defend against immediate threats (regardless of elixir if threat is severe)
        elif self.should_defend(enemy_troops):
            if self.current_elixir >= 3:  # Ensure we can defend
                return "DEFEND"
            elif self.elixir_urgency in ["HIGH", "MEDIUM"]:  # Use stored elixir for defense
                return "DEFEND"
        
        # Priority 3: High elixir leak prevention (9+ elixir)
        elif self.elixir_urgency == "HIGH":
            return "PREVENT_LEAK"
        
        # Priority 4: Counter-push with good elixir (6+ elixir)
        elif self.should_counter_push(our_troops, enemy_troops) and self.current_elixir >= 6:
            return "COUNTER_PUSH"
        
        # Priority 5: Medium elixir management (8 elixir)
        elif self.elixir_urgency == "MEDIUM":
            if enemy_troops:  # If enemies present, defend efficiently
                return "DEFEND"
            else:  # Otherwise apply pressure
                return "PRESSURE"
        
        # Priority 6: Apply pressure when safe (5-7 elixir, no immediate threats)
        elif self.should_pressure(self.current_elixir, enemy_troops) and self.current_elixir >= 5:
            return "PRESSURE"
        
        # Priority 7: Save elixir when low (2-4 elixir)
        elif self.elixir_urgency == "SAVE":
            if self.should_defend(enemy_troops):  # Only play if absolutely necessary
                return "DEFEND"
            else:
                return "WAIT"
        
        # Default: Wait for better opportunity or elixir
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
        
        # 🧠 RESET ADAPTIVE MEMORY FOR NEW GAME (keep long-term learning)
        self.cards_played_this_game = []
        self.positions_used_this_game = []
        self.my_last_card = None
        self.my_last_position = None
        self.my_last_play_time = time.time()
        
        # Keep opponent memory but reset recent tracking
        self.opponent_memory["last_detected_troops"] = []
        # Keep long-term patterns but reset variety counters for new game
        self.position_memory["position_variety_counter"] = {}
        self.position_memory["recent_failures"] = []
        
        print("🔄 Adaptive memory reset for new game")
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
        enemy_entities = getattr(self, "latest_enemy_entities", [])
        for entity in enemy_entities:
            ex_norm = entity.get("x", 0.0) / self.actions.WIDTH if self.actions.WIDTH else 0.0
            ey_norm = entity.get("y", 0.0) / self.actions.HEIGHT if self.actions.HEIGHT else 0.0
            opponent_troops.append({
                'x': entity.get("x", int(ex_norm * self.actions.WIDTH)),
                'y': entity.get("y", int(ey_norm * self.actions.HEIGHT)),
                'class': entity.get("class", "enemy_troop"),
                'confidence': entity.get("confidence", 0.0)
            })

        # Analyze opponent troops
        self.analyze_opponent_troops(opponent_troops)
        
        # 🧠 ADAPTIVE LEARNING - Track opponent responses
        self.track_opponent_response(opponent_troops)

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
        
        # 🔄 META-GAME STATE ANALYSIS
        self.analyze_meta_game_state(current_elixir, opponent_troops, our_troops)
        
        strategic_decision = self.get_strategic_decision(our_troops, opponent_troops)
        print(f"Strategic decision: {strategic_decision}")
        
        # ⚡ CONTEXTUAL ACTION SELECTION
        self.determine_action_context(strategic_decision, opponent_troops)
        high_level_strategy = self.select_high_level_strategy(strategic_decision)
        self.apply_real_time_adaptations(opponent_troops)
        
        # 🎲 ADVANCED AI SYSTEMS INTEGRATION
        # Update strategy patterns and risk tolerance based on performance
        if hasattr(self, 'objectives') and len(self.objectives.get('elixir_efficiency', [])) > 0:
            recent_efficiency = sum(self.objectives['elixir_efficiency'][-3:]) / min(len(self.objectives['elixir_efficiency']), 3)
            if recent_efficiency > 1.5:  # Good performance
                self.risk_assessment['risk_tolerance'] = min(self.risk_assessment['risk_tolerance'] * 1.05, 0.9)
                self.strategy_patterns['randomization_level'] = max(self.strategy_patterns['randomization_level'] * 0.95, 0.1)
            elif recent_efficiency < 0.8:  # Poor performance
                self.risk_assessment['risk_tolerance'] = max(self.risk_assessment['risk_tolerance'] * 0.95, 0.2)
                self.strategy_patterns['randomization_level'] = min(self.strategy_patterns['randomization_level'] * 1.05, 0.5)

        # Use strategic decision to override action if needed
        action = self.available_actions[action_index]
        card_index, x_frac, y_frac = action

        # INTELLIGENT CARD SELECTION - Override DQN choice if needed for better elixir management
        if card_index != -1 and card_index < len(self.current_cards):
            selected_card = self.current_cards[card_index]
            
            # 🎯 CONTEXTUAL CARD SELECTION - Consider game state, opponent patterns + elixir
            contextual_card = self.get_contextual_card_choice(self.current_cards, strategic_decision)
            if contextual_card and contextual_card != selected_card:
                # Find the index of the smarter card choice
                try:
                    contextual_card_index = self.current_cards.index(contextual_card)
                    card_cost = self.get_card_cost(contextual_card)
                    selected_cost = self.get_card_cost(selected_card)
                    
                    # Override if: 1) Contextual choice different, 2) Can't afford selected, 3) Better for context
                    should_override = (
                        contextual_card != selected_card and (
                            strategic_decision == "PREVENT_LEAK" or
                            selected_cost > self.current_elixir or
                            (self.elixir_urgency in ["HIGH", "CRITICAL"] and card_cost < selected_cost) or
                            self.action_context["forced_defense"] or  # Always override for forced defense
                            contextual_card != self.get_adaptive_card_choice(self.current_cards, strategic_decision)  # Contextual choice is different
                        )
                    )
                    
                    if should_override:
                        card_index = contextual_card_index
                        print(f"🎯 Contextual override: {selected_card} ({selected_cost}⚡) → {contextual_card} ({card_cost}⚡) [Mode: {self.action_context['mode']}]")
                        
                except ValueError:
                    pass  # Smart card not in current hand

        # Apply strategic positioning if we're playing a card
        if card_index != -1 and card_index < len(self.current_cards):
            card_name = self.current_cards[card_index]
            card_archetype = self.detect_card_archetype(card_name)
            card_cost = self.get_card_cost(card_name)
            
            # Final elixir check - don't play if we can't afford it
            if card_cost > self.current_elixir:
                print(f"❌ Cannot afford {card_name} ({card_cost}⚡) with {self.current_elixir}⚡ - skipping turn")
                # Return no-op state
                next_state = self._get_state()
                return next_state, -1, False  # Small penalty for bad elixir management
            
            # 📍 ADAPTIVE POSITIONING - Use success history and avoid failures
            strategic_positions = self.get_adaptive_positions(card_archetype)
            
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
            
            # 📊 TRACK PLAY FOR ADAPTIVE LEARNING
            self.my_last_card = card_name
            self.my_last_position = (x, y)
            self.my_last_play_time = time.time()
            self.cards_played_this_game.append(card_name)
            self.positions_used_this_game.append((x_frac, y_frac))
            
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
                # 🎯 Mark last position as successful (major success)
                if hasattr(self, 'my_last_position') and self.my_last_position:
                    normalized_pos = ((self.my_last_position[0] - self.actions.TOP_LEFT_X) / self.actions.WIDTH,
                                    (self.my_last_position[1] - self.actions.TOP_LEFT_Y) / self.actions.HEIGHT)
                    self.update_position_success(normalized_pos, was_successful=True)
                    print(f"🏆 TOWER DESTROYED! Position marked as highly successful")
        self.prev_enemy_princess_towers = current_enemy_princess_towers

        # 📍 ADAPTIVE LEARNING - Evaluate position success based on enemy reduction
        current_state = self._get_state()
        prev_state_for_objectives = self._get_state() if not hasattr(self, 'prev_state_snapshot') else self.prev_state_snapshot
        
        if (hasattr(self, 'my_last_position') and self.my_last_position and 
            hasattr(self, 'prev_enemy_presence') and self.prev_enemy_presence is not None):
            
            current_enemy_presence = sum(current_state[1 + 2 * MAX_ALLIES:][1::2]) if current_state is not None else 0
            enemy_reduction = self.prev_enemy_presence - current_enemy_presence
            
            # Evaluate if our last play was successful
            was_successful = enemy_reduction > 0.1  # Significant enemy reduction
            
            if hasattr(self, 'cards_played_this_game') and len(self.cards_played_this_game) > 0:
                normalized_pos = ((self.my_last_position[0] - self.actions.TOP_LEFT_X) / self.actions.WIDTH,
                                (self.my_last_position[1] - self.actions.TOP_LEFT_Y) / self.actions.HEIGHT)
                self.update_position_success(normalized_pos, was_successful)
        
        # Store state snapshot for next iteration
        self.prev_state_snapshot = current_state.copy() if current_state is not None else None
        
        done = False
        base_reward = self._compute_reward(current_state)
        
        # 🎯 ENHANCED MULTI-OBJECTIVE REWARD
        objective_bonus = 0
        if len(self.objectives["elixir_efficiency"]) > 0:
            avg_efficiency = sum(self.objectives["elixir_efficiency"]) / len(self.objectives["elixir_efficiency"])
            objective_bonus += avg_efficiency * 2  # Bonus for efficient trades
            
        objective_bonus += self.objectives["adaptation_score"] * 0.5  # Bonus for adapting to opponent
        objective_bonus += self.objectives["position_variety_score"] * 1  # Bonus for position variety
        
        total_reward = base_reward + spell_penalty + princess_tower_reward + objective_bonus
        
        # 🎯 MULTI-OBJECTIVE LEARNING
        if hasattr(self, 'my_last_card'):
            self.update_multi_objectives(prev_state_for_objectives, current_state, self.my_last_card, total_reward)
            
            # 🧩 UPDATE COMBO SUCCESS RATES
            if hasattr(self, 'combo_system') and self.combo_system.get('current_combo_setup'):
                combo_name = self.combo_system['current_combo_setup']
                # Simple success metric: positive reward = successful combo
                if total_reward > 0:
                    current_rate = self.combo_system['combo_success_rate'].get(combo_name, 0.5)
                    self.combo_system['combo_success_rate'][combo_name] = min(current_rate * 1.1, 1.0)
                    print(f"🧩 Combo '{combo_name}' success rate updated: {self.combo_system['combo_success_rate'][combo_name]:.2f}")
                else:
                    current_rate = self.combo_system['combo_success_rate'].get(combo_name, 0.5)
                    self.combo_system['combo_success_rate'][combo_name] = max(current_rate * 0.9, 0.1)
                    
                self.combo_system['current_combo_setup'] = None  # Reset after evaluation
        
        next_state = current_state
        return next_state, total_reward, done

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
            self.latest_enemy_entities = []
            self.latest_ally_entities = []
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

        allies = []
        enemies = []
        ally_entities = []
        enemy_entities = []

        for p in predictions:
            if not isinstance(p, dict):
                continue
            cls = normalize_class(p.get("class", ""))
            if cls in TOWER_CLASSES or "x" not in p or "y" not in p:
                continue
            entity = {
                "class": cls,
                "x": p["x"],
                "y": p["y"],
                "confidence": p.get("confidence", 0.0)
            }
            if cls.startswith("ally"):
                allies.append((p["x"], p["y"]))
                ally_entities.append(entity)
            elif cls.startswith("enemy") or cls:
                enemies.append((p["x"], p["y"]))
                enemy_entities.append(entity)

        print("Allies:", allies)
        print("Enemies:", enemies)

        # Persist latest entity information for downstream strategic modules
        self.latest_ally_entities = ally_entities
        self.latest_enemy_entities = enemy_entities

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

        # Build contextual feature vector for enhanced state representation
        context_features = []

        elixir_advantage = self.game_state.get("elixir_advantage", 0)
        context_features.append(float(np.clip(elixir_advantage / 10.0, -1.0, 1.0)))

        momentum_map = {"LOSING": -1.0, "NEUTRAL": 0.0, "WINNING": 1.0}
        context_features.append(momentum_map.get(self.game_state.get("momentum"), 0.0))

        phase_map = {"EARLY": 0.0, "MID": 0.33, "LATE": 0.66, "OVERTIME": 1.0}
        context_features.append(phase_map.get(self.game_state.get("game_phase"), 0.0))

        pressure_map = {"NONE": 0.0, "LOW": 0.33, "MEDIUM": 0.66, "HIGH": 1.0}
        context_features.append(pressure_map.get(self.game_state.get("pressure_level"), 0.0))

        context_features.append(float(np.clip(self.risk_assessment.get("risk_tolerance", 0.5), 0.0, 1.0)))
        context_features.append(float(np.clip(self.strategy_patterns.get("randomization_level", 0.3), 0.0, 1.0)))

        forced_defense = 1.0 if self.action_context.get("forced_defense") else 0.0
        context_features.append(forced_defense)

        aggression_map = {"DEFENSIVE": 0.0, "NEUTRAL": 0.5, "AGGRESSIVE": 1.0}
        context_features.append(aggression_map.get(self.opponent_model.get("aggression_prediction"), 0.5))

        context_features.append(float(np.clip(self.opponent_model.get("prediction_confidence", 0.0), 0.0, 1.0)))

        recent_reactions = self.opponent_memory.get("reaction_time", [])[-5:]
        avg_reaction = sum(recent_reactions) / len(recent_reactions) if recent_reactions else 0.0
        context_features.append(float(np.clip(avg_reaction / 10.0, 0.0, 1.0)))

        lane_pref = self.opponent_memory.get("preferred_lanes", {"left": 1.0, "right": 1.0, "center": 1.0})
        lane_total = sum(lane_pref.values()) or 1.0
        context_features.append(float(lane_pref.get("left", 0.0) / lane_total))
        context_features.append(float(lane_pref.get("right", 0.0) / lane_total))

        context_features.append(float(np.clip(self.meta_adaptation.get("meta_confidence", 0.0), 0.0, 1.0)))

        if self.combo_system.get("combo_success_rate"):
            best_combo_success = max(self.combo_system["combo_success_rate"].values())
        else:
            best_combo_success = 0.5
        context_features.append(float(np.clip(best_combo_success, 0.0, 1.0)))

        state_vector = [elixir / 10.0] + ally_flat + enemy_flat + context_features
        state = np.array(state_vector, dtype=np.float32)

        # Keep state size consistent with computed vector length
        if len(state) != self.state_size:
            self.state_size = len(state)
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

        # Enhanced Elixir Management Rewards
        elixir_reward = 0
        
        # Reward good elixir range (6-8 elixir is optimal)
        if 6 <= elixir <= 8:
            elixir_reward += 1
        elif elixir >= 9:
            elixir_reward -= 2  # Penalty for potential leaking
        elif elixir >= 10:
            elixir_reward -= 5  # Heavy penalty for full elixir
        elif elixir <= 2:
            elixir_reward -= 1  # Small penalty for being too low
            
        # Reward efficient elixir spending
        if self.prev_elixir is not None and self.prev_enemy_presence is not None:
            elixir_spent = self.prev_elixir - elixir
            enemy_reduced = self.prev_enemy_presence - enemy_presence
            
            # Reward elixir spending that reduces enemy presence
            if elixir_spent > 0 and enemy_reduced > 0:
                efficiency_bonus = 2 * min(elixir_spent, enemy_reduced)
                reward += efficiency_bonus
                
            # Bonus for preventing elixir leak while defending
            if elixir_spent > 0 and self.prev_elixir >= 9:
                reward += 3  # Bonus for using elixir when near full
                
            # Penalty for wasting elixir (spending without reducing enemy presence)
            if elixir_spent > 0 and enemy_reduced <= 0:
                reward -= elixir_spent * 0.5

        # Add elixir management component to total reward
        reward += elixir_reward
        
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
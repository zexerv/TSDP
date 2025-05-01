# planner.py
import numpy as np
import time
import config
from utils import normalize_angles # For potential use in objectives

print("Loading planner.py...")

# --- OMPL Imports ---
try:
    from ompl import base as ob
    from ompl import geometric as og
    config.OMPL_AVAILABLE = True
    print("Successfully imported OMPL.")
except ImportError:
    print("\n--- ERROR ---")
    print("Could not import OMPL. Please ensure OMPL Python bindings are installed.")
    print("Try: 'conda install -c conda-forge ompl'")
    print("Planning functionality will be disabled.")
    print("-------------\n")
    config.OMPL_AVAILABLE = False
except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"An unexpected error occurred during OMPL import: {e}")
    print("Planning functionality will be disabled.")
    print("-------------\n")
    config.OMPL_AVAILABLE = False

# --- OMPL State Validity Checker ---
if config.OMPL_AVAILABLE:
    class SimpleUR5eStateValidityChecker(ob.StateValidityChecker):
        """ Basic state validity checker: checks joint limits ONLY. """
        def __init__(self, si):
            super().__init__(si)
            self.si_ = si # Store SpaceInformation

        def isValid(self, state):
            q = np.array([state[i] for i in range(self.si_.getStateDimension())])
            lower_ok = np.all(q >= config.JOINT_LIMITS_MIN)
            if not lower_ok: return False
            upper_ok = np.all(q <= config.JOINT_LIMITS_MAX)
            if not upper_ok: return False
            # --- Placeholder for Collision Checking ---
            return True # Assume valid

        def check_collisions(self, q):
            # TODO: Implement collision detection
            return True # Assume collision-free

    # --- OMPL Optimization Objectives (Placeholders) ---
    class PathLengthObjective(ob.PathLengthOptimizationObjective):
        """ Standard path length objective. """
        def __init__(self, si):
            super().__init__(si)


# --- OMPL Planner Class ---
class OMPLPlanner:
    def __init__(self):
        """Initializes the OMPL planner environment."""
        self.si = None
        self.space = None
        if not config.OMPL_AVAILABLE:
            print("OMPL not available, planner cannot be initialized.")
            return
        self._setup_space_information()

    def _setup_space_information(self):
        """Sets up the OMPL state space and space information."""
        try:
            self.space = ob.RealVectorStateSpace(config.NUM_JOINTS)
            bounds = ob.RealVectorBounds(config.NUM_JOINTS)
            for i in range(config.NUM_JOINTS):
                bounds.setLow(i, config.JOINT_LIMITS_MIN[i])
                bounds.setHigh(i, config.JOINT_LIMITS_MAX[i])
            self.space.setBounds(bounds)
            self.si = ob.SpaceInformation(self.space)
            validity_checker = SimpleUR5eStateValidityChecker(self.si)
            self.si.setStateValidityChecker(validity_checker)
            self.si.setup()
            print("OMPL SpaceInformation setup complete.")
        except Exception as e:
            print(f"Error setting up OMPL SpaceInformation: {e}")
            self.si = None; self.space = None

    def _get_optimization_objective(self):
        """Creates the optimization objective based on config weights."""
        if self.si is None: return None
        # For now, let RRT* use its default path length objective.
        return None

    def plan(self, q_start_np, q_goal_np, planner_type="RRTConnect", time_limit=None):
        """
        Plans a path from q_start to q_goal using the specified OMPL planner.
        Uses the provided time_limit, falling back to the global config if None.

        Args:
            q_start_np (np.array): Start configuration (radians).
            q_goal_np (np.array): Goal configuration (radians).
            planner_type (str): Name of the OMPL planner (e.g., "RRTstar", "RRTConnect").
            time_limit (float, optional): Override planning time limit. Defaults to None.


        Returns:
            tuple: (path_np, path_cost)
                   path_np (np.array): Solution path (N x NUM_JOINTS), or None if failed.
                   path_cost (float): Cost of the path (length), or float('inf') if failed.
        """
        if not config.OMPL_AVAILABLE or self.si is None:
            return None, float('inf')

        # Use provided time limit or default from config
        current_planning_time = time_limit if time_limit is not None else config.PLANNING_TIME_LIMIT

        pdef = ob.ProblemDefinition(self.si)
        start_state = ob.State(self.space)
        goal_state = ob.State(self.space)
        for i in range(config.NUM_JOINTS):
            start_state[i] = q_start_np[i]
            goal_state[i] = q_goal_np[i]
        pdef.setStartAndGoalStates(start_state, goal_state)

        objective = self._get_optimization_objective()
        if objective: pdef.setOptimizationObjective(objective)

        try:
            planner_class = getattr(og, planner_type)
            planner = planner_class(self.si)
        except Exception as e:
            print(f"Error instantiating planner '{planner_type}': {e}")
            return None, float('inf')

        planner.setProblemDefinition(pdef)
        planner.setup()

        # Solve
        # start_time = time.time() # Timing moved to main loop if needed
        pdef.clearSolutionPaths()
        solved = planner.solve(current_planning_time)
        # planning_time = time.time() - start_time

        # Process Solution
        solution_path_np = None
        solution_cost = float('inf')

        if solved:
            if pdef.hasApproximateSolution():
                 print(f"  Warning: Solution is approximate. Goal distance: {pdef.getSolutionDifference():.4f}")
            path_geometric = pdef.getSolutionPath()
            # Simplify
            simplifier = og.PathSimplifier(self.si)
            try: simplifier.simplifyMax(path_geometric)
            except Exception as e: print(f"  Path simplification failed: {e}")
            # Interpolate
            if config.INTERPOLATE_PATH_POINTS > 0:
                try:
                    original_states = path_geometric.getStateCount()
                    if original_states > 1 and original_states < config.INTERPOLATE_PATH_POINTS:
                        path_geometric.interpolate(config.INTERPOLATE_PATH_POINTS)
                except Exception as e: print(f"  Path interpolation failed: {e}")
            # Cost
            solution_cost = path_geometric.length()
            # Extract
            solution_path_np = self._extract_path_numpy(path_geometric, verbose=False)

        return solution_path_np, solution_cost


    def _extract_path_numpy(self, path_geometric, verbose=True):
        """Extracts numpy arrays from an OMPL geometric path."""
        if path_geometric is None or path_geometric.getStateCount() == 0: return None
        if verbose: print(f"Extracting {path_geometric.getStateCount()} states from path...")
        path_np = np.array([
            [path_geometric.getState(i)[j] for j in range(self.si.getStateDimension())]
            for i in range(path_geometric.getStateCount())
        ])
        return path_np

print("planner.py loaded successfully.")

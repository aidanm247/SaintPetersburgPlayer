import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

public class SPMCTSPlayer extends SPPlayer { // simplified and ported from
    // Marc Lanctot's OpenSpiel MCTS implementation in C++

    double uctC = 2.0; // UCT exploration constant
    int numChanceSamples = 10; // Number of chance samples per chance node
    int numIterations = 1000000; // Number of MCTS iterations per move
    int playoutTerminationDepth = 4; // Depth at which to terminate playouts
    AiDanStateFeaturesLR1 features = new AiDanStateFeaturesLR1(); // Features for heuristic evaluation
    boolean verbose = true; // Verbosity flag
    Random chanceSeedRng = new java.util.Random(); // RNG for chance seeds
    int nodes = 0; // Node counter
    // time management fields
    private long startMs = UNKNOWN_TIME; // start time of move computation
    int endEstimatePlayouts = 3; // number of playouts to estimate the
    //  number of remaining decisions
    double fOpening = 1.1; // bias towards opening move search
    // Hendrik Baier and Mark H.M. Winands, "Time Management for Monte-Carlo
    //   Tree Search in Go" (ACG 2013)
    // see also their "Time Management for Monte-Carlo Tree Search" (2016)

    public SPMCTSPlayer() {
        super("SPMCTSPlayer");
    }

    class SearchNode {
        public int action = 0; // action index taken to get to this node
        public int player = 0; // player that acted to get to this node,
                          // or a negative seed value for a chance node
        public int exploreCount = 0; // number of times node was explored
        public double totalReward = 0.0; // sum of rewards from simulations
        public final List<SearchNode> children = new ArrayList<>(); // child nodes

        // constructor initializing action and player
        public SearchNode(int action, int player) {
            this.action = action;
            this.player = player;
        }

        public double uctValue(int parentExploreCount, double uctC) {
            if (exploreCount == 0) {
                return Double.POSITIVE_INFINITY; // prioritize unvisited nodes
            }
            double avgReward = totalReward / exploreCount;
            double explorationTerm = uctC * Math.sqrt(Math.log(parentExploreCount) / exploreCount);
            return avgReward + explorationTerm;
        }

        public SearchNode bestChildByVisits() {
            return children.stream()
                    .max((a, b) -> Integer.compare(a.exploreCount, b.exploreCount))
                    .orElse(null);
        }

        public SearchNode bestChildUCT() {
            return children.stream()
                    .max((a, b) -> Double.compare(a.uctValue(exploreCount, uctC),
                            b.uctValue(exploreCount, uctC)))
                    .orElse(null);
        }

        public String toString() {
            return String.format("%d: Player: %d, Visits: %d, Total Reward: %f, Q: %.2f, Children: %d",
                    action, player, exploreCount, totalReward,
                    (exploreCount == 0 ? 0.0 : totalReward / exploreCount),
                    children.size()
            );
        }

        public String childrenStr(int parentExploreCount, double uctC) {
            StringBuilder sb = new StringBuilder();
            Comparator<SearchNode> uctComparator = (a, b) -> Double.compare(
                    a.uctValue(parentExploreCount, uctC),
                    b.uctValue(parentExploreCount, uctC)
            );
            children.stream()
                    .sorted(uctComparator.reversed())
                    .forEach(child -> {
                        sb.append(String.format("%s\n", child.toString()));
                    });
            return sb.toString();
        }

        public boolean isChanceNode() {
            return player < 0;
        }   
    }

    @Override
    public int getAction(SPState state) {
        startMs = System.currentTimeMillis();
        // get the legal actions for the current state,
        // compute the number of legal actions,
        // call MCTSSearch to get the root SearchNode,
        // select the best child by visits,
        // and return the action of that child.
        ArrayList<SPAction> legalActions = state.getLegalActions();
        int numLegalActions = legalActions.size();
        SearchNode root = MCTSSearch(state);
        SearchNode bestChild = root.bestChildByVisits();
        if (verbose) {
            System.out.println("Number of legal actions: " + numLegalActions);
            System.out.println("Root Node:\n" + root.toString());
            System.out.println("Children:\n" + root.childrenStr(root.exploreCount, uctC));
            System.out.println("Selected Action: " + bestChild.action);
        }
        return bestChild.action;
    }

    public SearchNode MCTSSearch(SPState rootState) { // UCT_SEARCH
        long turnSearchTimeMillis = 1000L; // default fixed time per move
        if (timeRemainingMillis != UNKNOWN_TIME) {
            // Estimate the number of decisions remaining (including this)
            // For this we do a specified number of playouts, count the
            //   current player decisions, and average.
            int currentPlayer = rootState.playerTurn;
            int totalDecisions = 0;
            for (int p = 0; p < endEstimatePlayouts; p++) {
                SPState simState = rootState.clone();
                while (!simState.isGameOver()) {
                    if (simState.playerTurn == currentPlayer) {
                        totalDecisions++;
                    }
                    ArrayList<SPAction> legalActions = simState.getLegalActions();
                    int actionIndex = (int) (Math.random() * legalActions.size());
                    SPAction action = legalActions.get(actionIndex);
                    simState = action.take();
                }
            }
            double movesExpected = (double) totalDecisions / endEstimatePlayouts;
            if (verbose) { // print the estimated number of decisions remaining
                System.out.printf("Estimated decisions remaining: %.2f\n", movesExpected);
            }
            // Allocate time for this move
            turnSearchTimeMillis = (long) (fOpening * timeRemainingMillis / movesExpected);
            // Ensure that the move time is not more than a 20th of the remaining time
            turnSearchTimeMillis = Math.min(turnSearchTimeMillis, 
                timeRemainingMillis / 20L);
        }
        // Define variables to support periodic checking of the clock
        int numBlockIterations = 100; // check time every 100 iterations
        long lastIterationBlockMillis = UNKNOWN_TIME; // time since last check
        long blockStartMillis = System.currentTimeMillis();

        SearchNode rootNode = new SearchNode(0, rootState.playerTurn);
        expand(rootNode, rootState);
        nodes = 1; // reset node counter
        startMs = System.currentTimeMillis();

        List<SearchNode> path = new ArrayList<>(); // store sequence of
                                                // SearchNodes visited
        for (int iter = 0; iter < numIterations; iter++) { // MCTS loop
            // Check elapsed time every numBlockIterations
            if ((iter + 1) % numBlockIterations == 0) {
                long currentMillis = System.currentTimeMillis();
                lastIterationBlockMillis = currentMillis - blockStartMillis;
                long elapsedMillis = currentMillis - startMs;
                // If another block of the same duration would exceed
                // the allocated time, break
                if (elapsedMillis + lastIterationBlockMillis > turnSearchTimeMillis) {
                    if (verbose) {
                        System.out.printf("MCTS terminating at iteration %d due to time limit.\n", iter + 1);
                    }
                    break;
                }
                blockStartMillis = currentMillis;
            }

            SPState state = rootState.clone();
            path.clear();
            SearchNode node = rootNode;

            // Selection/Expansion phase (TREE_POLICY)
            // While the state is non-terminal:
            // - Add the non-chance node to the path
            // - If the node is not fully expanded, expand it and break
            // - Otherwise, select the best child by UCT 
            //   (prioritizing unselected children).
            // - If the action is a chance action, expand the chance node
            //   if needed, and reproduce a chance-sampled outcome using
            //   the stored seed + action index.
            // We assume that chance nodes are not followed by chance nodes.
            // After this phase, the path will contain all non-terminal
            //   nodes visited, and we will either be at a terminal state
            //   have just left the search tree.

            while (!state.isGameOver()) {
                path.add(node);
                if (node.children.isEmpty()) {
                    // Node is not expanded
                    expand(node, state); // expand and break
                    break;
                }
                // Node is expanded; select best child by UCT
                SearchNode nextNode = node.bestChildUCT();
                if (nextNode.isChanceNode()) {
                    path.add(nextNode); // add chance node to path
                    // Chance node: reproduce chance outcome
                    SPAction chanceAction = state.getLegalActions().get(nextNode.action);
                    // Use stored seed + action index to seed RNG
                    // Sample one of the chance outcomes
                    int sampleIndex = (int) (Math.random() * numChanceSamples);
                    int sampleSeed = nextNode.player + sampleIndex;
                    state = chanceAction.take(sampleSeed);
                    node = nextNode.children.get(sampleIndex);
                } else {
                    // Non-chance action node
                    state = state.getLegalActions().get(nextNode.action).take();
                    node = nextNode;
                }
            }
            
            // Simulation phase (DEFAULT_POLICY) and Evaluation
            // Early playout termination (EPT) after a fixed depth
            int stepsRemaining = playoutTerminationDepth;
            while (!state.isGameOver() && stepsRemaining > 0) {
                ArrayList<SPAction> legalActions = state.getLegalActions();
                int actionIndex = (int) (Math.random() * legalActions.size());
                SPAction action = legalActions.get(actionIndex);
                state = action.take();
                stepsRemaining--;
            }

            // Evaluate the (possibly non-terminal) state
            double[] returns = new double[state.numPlayers];

            // NOTE: This assumes win probability, but would need to be
            // modified for score difference
            if (state.isGameOver()) {
                // Terminal state: use actual returns
                for (int i = 0; i < state.numPlayers; i++) {
                    returns[i] = state.isWinner[i] ? 1.0 : 0.0;
                }
            } else {
                // Non-terminal state: use heuristic evaluation
                double winProb = features.predict(state);
                if (state.playerTurn == 0) {
                    returns[0] = winProb;
                    returns[1] = 1.0 - winProb;
                } else {
                    returns[0] = 1.0 - winProb;
                    returns[1] = winProb;
                }
                
            }

            // Backpropagation phase (BACKUP)
           for (int i = path.size() - 1; i >= 0; i--) {
                SearchNode n = path.get(i);
                n.exploreCount += 1;
                if (n.player >= 0) {
                    // Non-chance node: back up the return for that player
                    n.totalReward += returns[n.player];
                }
                else {
                    // Chance node: acting player is in the child nodes
                    int actingPlayer = path.get(i + 1).player;
                    n.totalReward += returns[actingPlayer];
                }
            }
        }

        long endMillis = System.currentTimeMillis();
        if (verbose) {
            System.out.printf("MCTS completed in %d ms, %d nodes created.\n",
                    (endMillis - startMs), nodes);
            // Print the root node and its children as well as the selected best child
            System.out.println("Root Node:\n" + rootNode.toString());
            System.out.println("Children:\n" + rootNode.childrenStr(rootNode.exploreCount, uctC));
        }
    
        return rootNode;
    }

    public void expand(SearchNode node, SPState state) {
        // Assumes state is non-terminal, non-chance
        // For each non-chance action, add a child SearchNode.
        // For each chance action, create an expanded chance-sampling
        //   SearchNode with a negative "player" value for reproducable
        //   chance-sampling, and children that reflect the player
        //   that chose the chance action.

        ArrayList<SPAction> legalActions = state.getLegalActions();

        // Create a random permutation of the action indices
        // in int array actionIndices
        int numLegalActions = legalActions.size();
        int[] actionIndices = new int[numLegalActions];
        for (int i = 0; i < numLegalActions; i++) {
            actionIndices[i] = i;
        }
        // Fisher-Yates shuffle (really Durstenfeld's shuffle 1964
        // and later Knuth shuffle 1969)
        for (int i = numLegalActions - 1; i > 0; i--) {
            int j = (int) (Math.random() * (i + 1));
            int temp = actionIndices[i];
            actionIndices[i] = actionIndices[j];
            actionIndices[j] = temp;
        }

        for (int a : actionIndices) {
            SPAction action = legalActions.get(a);
            int player = state.playerTurn;
            if (action.isChanceAction()) {
                // Create chance-sampling node
                int chanceSeed = -(chanceSeedRng.nextInt(Integer.MAX_VALUE - numChanceSamples) + numChanceSamples);
                SearchNode chanceNode = new SearchNode(a, chanceSeed);
                node.children.add(chanceNode);
                nodes++;

                // Create children for each chance outcome
                for (int sample = 0; sample < numChanceSamples; sample++) {
                    SearchNode childNode = new SearchNode(a, player);
                    chanceNode.children.add(childNode);
                    nodes++;
                }
            } else {
                // Non-chance action node
                SearchNode childNode = new SearchNode(a, player);
                node.children.add(childNode);
                nodes++;
            }
        }

        
    }


}

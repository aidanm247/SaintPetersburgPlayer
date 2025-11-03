import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

public class SPMCTSPlayer extends SPPlayer { // simplified and ported from
    // Marc Lanctot's OpenSpiel MCTS implementation in C++

    double uctC = 2.0; // UCT exploration constant
    int numChanceSamples = 10; // Number of chance samples per chance node
    int numIterations = 20000; // Number of MCTS iterations per move
    int playoutTerminationDepth = 4; // Depth at which to terminate playouts
    SPStateFeaturesLR1 features = new SPStateFeaturesLR1(); // Features for heuristic evaluation
    boolean verbose = true; // Verbosity flag
    Random chanceSeedRng = new java.util.Random(); // RNG for chance seeds
    int nodes = 0; // Node counter

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
            return String.format("%d: Player: %d, Visits: %d, Q: %.2f, Children: %d",
                    action, player, exploreCount, 
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
        SearchNode rootNode = new SearchNode(0, rootState.playerTurn);
        expand(rootNode, rootState);
        nodes = 1; // reset node counter
        long startMillis = System.currentTimeMillis();

        List<SearchNode> path = new ArrayList<>(); // store sequence of
                                                // SearchNodes visited
        for (int iter = 0; iter < numIterations; iter++) {
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
                    int sampleSeed = -nextNode.player + sampleIndex;
                    state = chanceAction.take(sampleSeed);
                    node = nextNode.children.get(sampleIndex);
                } else {
                    // Regular action node
                    // TODO
                }
            }

            // Expansion
            if (!state.isTerminal()) {
                expand(node, state);
                // Choose one of the new children to continue
                SearchNode nextNode = node.bestChildUCT();
                path.add(nextNode);
                SPAction action = state.getLegalActions().get(nextNode.action);
                state.applyAction(action);
                node = nextNode;
            }

            // Simulation
            double reward = SPPlayouts.simulatePlayout(state, playoutTerminationDepth, features);

            // Backpropagation
            for (SearchNode visitedNode : path) {
                visitedNode.exploreCount += 1;
                visitedNode.totalReward += reward;
            }
        }


        if (verbose) {
            System.out.println("Total nodes created: " + nodes);
            System.out.println("Total seconds for MCTS: " +
                    ((System.currentTimeMillis() - startMillis) / 1000.0)); 
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
                // Regular action node
                SearchNode childNode = new SearchNode(a, player);
                node.children.add(childNode);
                nodes++;
            }
        }

    }


}

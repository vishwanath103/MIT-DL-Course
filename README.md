# MIT Deep Learning Crash Course
## Lecture 4: Deep Generative Modeling : Notes
* Unsupervised Learning
  * **Goal:** Learn the hidden or underlying structure of the data
* Generative Modeling
  * **Goal:** Take as input training samples from some distribution and learn a model that represents that distribution
* Why generative models?
  * **Debiasing:** Capable of uncovering underlying features in a dataset
  * **Outlier detection**
* Latent variable models
  *  Autoencoders and Variational Autoencoders (VAEs)
  *  Generative Adversarial Networks (GANs)
* What is latent variable?
  * Can we learn the true explanatory factors
* Autoencoders: background
  * Unsupervised approach for learning a lower-dimensional feature representation from unlabeled training data
  * "Encoder" learns mapping from the data, x, to a low-dimensional latent space, z
  * To learn from this latent space, train the model to use the features to reconstruct the original data
  * "Decoder" learns mapping back from latent space,z, to a reconstructed observation, x^
  *  L(x,x^) = ||x-x^||<sup>2</sup>
* Dimensionality of latent space --> reconstruction quality
  * Autoencoding is a form of compression
  * Smaller latent space will force a larger training bottleneck
* Autoencoders for representation learning
  * **Bottleneck hidden layer** forces the network to learn a compressed latent representation
  * **Reconstruction loss** forces the latent representation to capture (or encode) as much information about the data as possible
* VAEs
  * Variational autoencoders are a probabilistic twist on autoencoders
  * Sample from the mean and standard deviation to compute latent sample
  * Encoders and Decoders are probabilistic in nature
  * Loss = (reconstruction loss) + (regularization term)
  * What properties do we want to achieve from regularization?
    1.  **Continuity:** points that are close in latent space --> similar content after decoding
    2.  **Completeness:** sampling from latent space --> meaningful content after decoding
  * Reparametrizing the sampling layer for backpropagation
  * Latent space disentanglement with beta-VAEs
  * VAE Summary
    1.  Compress representation of world to something we can use to learn
    2.  Reconstruction allows for unsupervised learning
    3.  Reparameterization trick to train ene-to-end
    4.  Interpret hidden latent variables using perturbation
    5.  Generating new examples
* Generative Adversarial Networks (GAN)
  * Sample from something simple (eg: noise), learn a transformation to the data distribution
  * GAN are a way to make a generative model by having two neural networks compete with each other
    * The **generator** turns noise into an imitation of the data to try to trick the discriminator
    * The **discriminator** tries to identify real data from fakes created by the generator
  * CycleGAN: domain transformation
    * data manifold X to data manifold Y
    * Transforming speech
  
 ## Lecture 5: Reinforcement Learning : Notes
  * **Data:** state-action pairs
  * **Goal:** Maximize future rewards over many time steps
  * RL: Key Concepts
    * **AGENT:** takes actions.
    * **ENVIRONMENT:** the world in which the agent exists and operates
    * **Action:** a move the agent can make in the environment.
    * **Action Space:** the set of possible actions an agent can make in the evironment
    * **Observations:** of the environment after taking actions.
    * **State:** a situation which the agent perceives
    * **Reward:** feebback that measures the success or failure of the agents action
    * Total Reward and Discounted Total Reward
    * **Q-function:** captures the expected total future reward an agent in state s can receive by executing a certain action a
  * How to take actions given a Q-function?
    * The agent needs a policy, to infer the best action to take at its state,s
    * **Strategy:** the policy should choose an action that maximizes the future reward
  * Deep Reinforcement Learning Algorithms
    * Value Learning: Q Learning
    * Policy Learning; Phi learning
  * Deep Q Networks
    * ![image](https://user-images.githubusercontent.com/6136144/111029063-a2cb8980-8420-11eb-8ffb-b8726fb86ac4.png)
  * Downsides of Q-Learning
    * **Complexity:** Cannot handle continuous action spaces
    * **Flexibility:** Cannot learn stochastic policies
  * Policy Learning
    * Directly optimize the policy
    * **Policy Gradient:** Enables modeling of continuous action space, by outputing the mean and SD of the probability distribution
  * Training Policy Gradients
    * Intialize the agent
    * Run a policy until termination
    * Record all states, actions, rewards
    * Decrease probability of actions that resulted in low reward
    * Increase probability of actions that resulted in high reward

## Lecture 6: Deep Learning New Frontiers : Notes
* Power of Neural Nets
 * Universal Approximation Theorem
* Limitations
 * Rethinking Generalization: "Understanding Deep Neural Networks Requires Rethinking Generalization"
 * Neural Networks as Function Approximators
 * Deep Learning = Alchemy? --> Garbage in and Garbage out
 * UNcertainity in Deep Learning

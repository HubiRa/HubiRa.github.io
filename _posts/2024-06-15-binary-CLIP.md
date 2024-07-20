---
title: "S-BRaCE"
author: hubi
date: 2024-05-15 21:40:00 +0200
description: Learning Sparse Binary Representations of Contrastive Embeddings
categories: [Blogging, ML]
tags: [
    machine learning,
    deep learning,
    contrastive learning,
    CLIP,
    binary CLIP,
    binarization
  ] # TAG names should always be lowercase
math: true
---

TODO: Revisit formulations (partly bad english)

> Accompanying code can be found here **\<in preparation\>**
> {: .prompt-info }
> (Even if the code is not available yet, the method is very simple and can be reproduced very easily. Nonetheless I will try to clean up the code and put it on GitHub asap.)

> This blog post is work in progress. Important changes will be added to the change log below
> {: .prompt-info }

<details>
    <summary>Change log</summary>
    <table>
        <tr>
            <th>Date</th>
            <th>Change</th>
        </tr>
        <tr>
            <td> None </td>
            <td> None </td>
        </tr>
    </table> 
</details>

## Intro

This blog post introduces a method called **S-BRaCE** to binarize CLIP[^1] (or other contrastively learned) embeddings.
If somebody wonders why we would even want to do that, answers might be: lower memory usage, potentially faster lookup, interpretability (more on that in section: "Zero-Shot Results")...however I am not sure if this are good answers (probably something to explore in follow up posts).

To be honest, this method originated just out of interest. Binarized CLIP[^1] embeddings might or might not be useful - at least to me it is an interesting problem. How would one train such a system? How well does it work? What limitations does it have? I believe that the general method which is developed here is not the worst approach and potentially could be of use for other problems (e.g. in binary optimization, as novel target representation etc. - i am exploring some currently). We will see.

### Prerequisite

It is assumed that the reader is familiar with [CLIP](https://arxiv.org/pdf/2103.00020)[^1] (and related methods like [CLOOB](https://arxiv.org/abs/2110.11316)[^2] and [SigLIP](https://arxiv.org/pdf/2303.15343)[^3])

> In the following, the term “CLIP[^1]” is used as a stand-in for other contrastive methods. “CLIP[^1]” is chosen for its familiarity.
> {: .prompt-info }

A more detailed literature review that also covers binary representation learning is planned!

## **S-BRaCE**

### The Idea

First, we make a conceivably simple observation: that binary vectors are corners of the unit hypercube. Let's assume we are in $R^3$, then we have (unsurprisingly) the following corners (excluding the origin):

$$
\begin{equation}
  \mathcal{B}_3 :=  \{0,1\}^3 \setminus \mathbf{0} =
 \left\{
  \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix},
  \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix},
  \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix},
  \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix},
  \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix},
  \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix},
  \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}
  \right\}
\end{equation}
$$

![alt text](/assets/img/plots/unit_hyper_cube2.png "Title")

<div style="border: 4px solid; border-color: #fffff; padding: 10px; background-color: #212322; color: #fffff; font-size: 1.2em;" >
  <p style="text-align: center;">The idea consists of two parts:</p>
  <ul>
    <li>a) learn to map CLIP embeddings into the unit hypercube via small FCNs and</li>
    <li>b) project the new embeddings onto the closest vectors in $\mathcal{B}_D$</li>
  </ul>
</div>

Here only b) is somewhat challenging. There is a surprisingly simple solution however. But more on that later. First we will start with a).

### a) Mapping Embeddings into the Unit Hypercube

The first task is to learn embeddings which lie somewhere inside of the unite hypercube. For this we can use any contrastive loss (for the experiments described we will use the usual CLIP[^1] loss). Formally, we want two properties for a resulting embedding $\mathbf{v}$:

$$
\begin{equation}
    \label{eq:positive-embs}
    \mathbf{v} \in [0, 1]^D
\end{equation}
$$

and

$$
\begin{equation}
    \label{eq:unit_norm}
    ||\mathbf{v}|| = 1
\end{equation}
$$

In order to achieve (\ref{eq:positive-embs}) we can learn a small FCN $f_{\mu}$ for each modew  $\mu$. By using the softplus activation function as output activation we ensure positive outputs:

$$
\begin{equation}
    \label{eq:softplus}
    \text{softplus}(x) = \text{log} (1 + \exp^x)
\end{equation}
$$

<img src="/assets/img/plots/softplus.png" alt="alt drawing" width="400">

This has of course the consequence that the inner product of two new embeddings $\mathbf{x}$ and $\mathbf{y}$ now is positive as well: $\mathbf{x}^T \mathbf{y} \in [0, 1]$. Thus a inner product of $0$ indicates maximal dissimilarity. This is in line with binary embeddings, where orthogonality means dissimilarity. So far so good. But given an output embedding, how do we assign a binary vector?

### b) Finding the right Projection

Lets assume we have trained our FCNs to our satisfaction and apply it on some data. We receive an embedding vector $\mathbf{v} \in [0, 1]^D$ and need to assign it to the "most similar" binary vector. We will do the following:

1. generate vectors of norm 1 which represent the binary vectors (referred to as **proxy** $\mathcal{U}_D$ for $\mathcal{B}_D$)
2. assign the $\mathcal{v}$ to the closest $ \mathcal{u} \in \mathcal{U}\_D$ in terms of some distance/similarity measure.

#### A Proxy for $\mathcal{B}_D$

A useful proxy can be generated by normalizing all vectors in $\mathcal{B}_D$. This will yield the following set:

$$
\begin{equation}
  \mathcal{U}_D :=
 \left\{
  { \frac{1}{ \sqrt{\mathbf{b}^T \mathbf{1}}} \mathbf{b}   \mid \mathbf{b} \in \mathcal{B}_D}
  \right\}
\end{equation}
$$

<img src="/assets/img/plots/proxie.png" alt="alt drawing" width="500">

Here, $\mathbf{b}^T \mathbf{1}$ counts the number of none-zero entries in $\mathbf{b}$. And since a binary vector with $n$ none-zero entries has length $\sqrt{n}$ we just need to divide $\mathbf{b}$ by $\sqrt{\mathbf{b}^T \mathbf{1}}$ to receive a vector with norm 1.

#### The optimization problem

For a given embedding $\mathbf{v} \in [0, 1]^D$ we now want to assign the closest vector in $\mathcal{U}_D$. Since all involved vectors are of unit norm it does not matter if we write the optimization problem as minimizing the difference in squared l2 norm or maximizing cosine similarity:

$$
\begin{equation}
    \label{eq:optim}
    \underset{ \mathbf{u} \in \mathcal{u}_D}{\operatorname{argmin}} || \mathbf{v} - \mathbf{u} ||^2,
\end{equation}
$$

where expanding the squared norm results in

$$
\begin{equation}
    \label{eq:sq_norm}
    || \mathbf{v} - \mathbf{u} ||^2 = ||\mathbf{v}||^2 - 2 \mathbf{v}^T\mathbf{u} + ||\mathbf{u}||^2 = ~ 2 - 2 ~\mathbf{v}^T\mathbf{u},
\end{equation}
$$

and thus

$$
\begin{equation}
    \label{eq:optim_cos }
    \underset{ \mathbf{u} \in \mathcal{U}_D}{\operatorname{argmin}} || \mathbf{v} - \mathbf{u} ||^2 =  \underset{ \mathbf{u} \in \mathcal{U}_D}{\operatorname{argmax}} ~ \mathbf{v}^T\mathbf{u}.
\end{equation}
$$

If we rewrite the problem in terms of $\mathbf{b}$ we get:

$$
\begin{equation}
    \label{eq:optim_step1}
    \underset{ \mathbf{b} \in \mathcal{B}_D}{\operatorname{argmax}} ~  \frac{1}{\sqrt{\mathbf{b}^T \mathbf{1}}} ~ \mathbf{v}^T\mathbf{b}.
\end{equation}
$$

Remember, that $\mathbf{b}$ is binary. So (\ref{eq:optim_step1}) tells us, that a solution to the optimization problem will

- select as few as possible entries of $\mathbf{v}$ since $ \frac{1}{\sqrt{\mathbf{b}^T \mathbf{1}}}$ should be as large as possible (sparseness!)
- select the largest entries of $\mathbf{v}$

Fortunately, the gods of machine learning are kind to us and we can simply write down a function that helps us find this optimum.

#### Finding the solution: a partial sum for complete optimization

But first notation: we take inspiration from programming languages, denote the function that returns the indices of the sorted (descending) entries of a vector as "$\text{argsort}$" and introduce the indexing operator $[\cdot]$. We apply this notation to get the following:

$$
\begin{equation}
    \label{eq:argsort}
    \mathbf{p} =  \text{argsort}(\mathbf{v})
\end{equation}
$$

and denote the sorted vector as

$$
\begin{equation}
    \label{eq:sort }
    \mathbf{v}[\mathbf{p}].
\end{equation}
$$

Now, for $0 < i < j \leq D $ we have:

$$
\begin{equation}
    \label{eq:sorted }
    \mathbf{v}[\mathbf{p}]_i ~ \geq ~ \mathbf{v}[\mathbf{p}]_j ~ \geq ~ \mathbf{0}.
\end{equation}
$$

We define a function such that finding its maximum also yields the optimal binary vector. This function is the following partial sum:

$$
\begin{equation}
    \label{eq:partial_sum}
    \mathcal{S}(K; \mathbf{v}) = \frac{1}{\sqrt{K}} \sum_{k=1}^K \mathbf{v}[\mathbf{p}]_k.
\end{equation}
$$

If we find the $K^\ast$ that optimizes this sum we are done: these $K^\ast$ entries of the index vector $\mathbf{p}$ tell us exactly which entries of the binary vector need to be one (the rest is zero). Formulated differently, these $K^\ast$ entries of $\mathbf{p}$ contain the indices of the optimal binary vector $\mathbf{b}^*$ that are $1$:

$$
\begin{equation}
    \label{eq:optimum}
    \mathbf{b}^*_{\mathbf{p}[1]}, \ldots, \mathbf{b}^*_{\mathbf{p}[K^*]} = 1  \quad and \quad   {\mathbf{b}^*}^T \mathbf{1} = K^*; \quad \mathbf{b}^* \in \mathcal{B}
    %\ast{\mathbf{b}}\mathbf{p}[:K]] = 1 \quad and \quad   {\mathbf{b}^*}^T \mathbf{1} = K
\end{equation}
$$

<div style="border: 1px solid gray; padding: 10px; background-color: #2b2925; font-size: 1.2em;">
    <p style="text-align: center;">Solution to (\ref{eq:optim_step1})</p>
    <ul>
        <li>In order to find the optimum we calculate all partial sums and select $K^* = \underset{K}{\operatorname{argmax}} \mathcal{S}(K)$</li>
        <li>Given $K^*$ The optimal binary vector $\mathbf{b}^* \in \mathcal{B_D}$ is the the one described in (\ref{eq:optimum})</li>
    </ul>
</div>

Lets plot $\mathcal{S}(K)$ for some learned embeddings after the softplus, normed to 1:

<img src="/assets/img/plots/SK_img.png" alt="alt drawing" width="500">

Each maximum in the above plot is a $K^\ast$ specific to one vector and represents a solution to the optimization problem posed in (\ref{eq:optim_step1}). We can already see that $K^\ast$ tend to be small, hinting at very sparse solutions. Additionally, this suggests that the embeddings are well distributed within the unit hypersphere and, consequently, the hypercube. It also indicates how effectively S-BRaCE has learned to transfer the embeddings into the positive hypersphere.
(Given embeddings from various methods (e.g., CLIP vs. CLOOB), **S-BRaCE** could be used to compare their quality. Assuming all other factors are equal, if **S-BRaCE** performs better with one method over another, it could indicate its superiority. Besides retrieval capabilities, sparseness and entropy of the binary representations could be indicators.)

This plot would suggest that we do not need to calculate all partial sums up until $D$, but only until we have found a $K$ such that $\mathcal{S}(K) > \mathcal{S}(K+1)$. Unfortunately we can find vectors $\hat{\mathbf{v}}$ such that $\mathcal{S}(K; \hat{\mathbf{v}})$ has a minimum as extremum. More on that in the Appendix.

### Comparing binary representations

After calculating binary representations we of course want to compare them - we need a similarity function. The obvious choice is the Jaccard Index (or IoU):

$$
\begin{equation}

    \mathcal{J}(\mathbf{x}, \mathbf{y}) = \frac{ \mathbf{x}^T \mathbf{y}}{ \mathbf{x}^T  \mathbf{1} + \mathbf{y}^T  \mathbf{1} - \mathbf{x}^T \mathbf{y}}; \quad \mathbf{x}, \mathbf{y} \in \mathcal{B}_D.
\end{equation}
$$

$\mathcal{J}(\mathbf{x}, \mathbf{y}) = 1$ if the exact same entries are active in both $\mathbf{x}$ and $\mathbf{y}$ and $0$ if no entries match. Importantly the similarity is downweighted if one of the binary representations has many more active entries then the other.

#### Why is the hamming distance not used?

The Hamming distance counts the different entries between two binary vectors (in our case). It would be simple to change the Hamming distance $h(\cdot)$ to a hamming similarity $h_s(\cdot) = 1 - h(\cdot)/D$. To see why the Hamming distance (or similarity) is not suitable for our case lets consider two vectors:

$$
    \label{eq:hamming-vec}
    \mathbf{v}_1 = [1, 0, 0, 0], \quad \mathbf{v}_2 = [0, 0, 0, 1]
$$

It is clear that for the Jaccard index we have $\mathcal{J}(\mathbf{v}_1, \mathbf{v}_2) = 0$. This is what we would expect, since there are two completely different "concepts" active (e.g. one could be dog, the other car). However, the Hamming distance would be $h(\mathbf{v}_1, \mathbf{v}_2) = 2$ and the Hamming similarity would be $h_s(\mathbf{v}_1, \mathbf{v}_2) = 0.5$ since half of the entries are the same. The larger $D$ the more severe this issue becomes. For $D = 256$ like in our experiments below, in this case we would get a Hamming similarity of $\approx 0.99$ which is clearly not what we want.

### A first Experiment

> Experimentation is expansive time wise and I've got to juggle my time between family and work first. It is planned that updates with more extensive and thorough experiments are added over time (first ones are already in preparation). The character of the current results is more that of sanity testing
> {: .prompt-info }

It is clear that having binary representations is limiting. The question is how limiting it is. Here, first results are reported for the following setting:

- The model is trained on a small subset of the [Conceptual captions dataset](https://ai.google.com/research/ConceptualCaptions/)[^4] (210K train, 3k validation)
- CLIP[^1] embeddings are precalculated where [**CLIP-vit-base-patch32**](https://huggingface.co/openai/clip-vit-base-patch32) is used
- The binary representation size is chosen to be 256 (the CLIP[^1] embeddings have a dimension of 512)
- No hyper parameter search is conducted for now. The training is just kicked of and we see what happens
  - Edit: Since the above line was written, I did try out a few more settings - but again not systematic. Rather to get a feeling for the method. More on that in the Appendix.

#### Some preliminary results and analysis

In a first run we train

- with two FCN's with input dim 512, output dim 256, one hidden layers with dimension 256 and the **GeLU**[^5] activation function
- with **AdamW**[^6] [^7] with initial learning rate of 1e-2
- using **StepLR** scheduler which decreases the learning rate at a factor of 0.9 each epoch
- for 20 epochs
- with a batch size of 256

The loss curve shows that **S-BRaCE** is learning; however, it starts to over-fit early and remains at a high loss level. There are many nuts and bolts to adjust to improve training. Here, we didn't even use early stopping, i.e. the reported results are from epoch 20. Likely, the most effective measure would be to train with more data.

<img src="/assets/img/plots/loss.png" alt="alt drawing" width="500">

A second simple verification if **S-BRaCE** does something useful is the following plot: We sample 250 image-text pairs from the validation dataset, embed them and calculate the Jaccard index for each pair. We then randomly shuffle the text embeddings and again calculate the Jaccard index. Then we sort both, the similarity of the matching and random pairs along the matching similarities. The similarities (Jaccard indices) of the matching pairs should be higher then the radom pairs. The green line indicates if the matching pair has higher similarity then the random pair. A green spike towards 0 indicates the latter.

<img src="/assets/img/plots/matching_vs_radon.png" alt="alt drawing" width="500">

So far so good. Next we will do a small qualitative investigation. I use [ChatGPT4o](https://openai.com/index/hello-gpt-4o/) to generate a image of an golden retriever with blue background.

<img src="/assets/img/plots/golden.webp" alt="alt drawing" width="300">

First I embed the image with both, **S-BRaCE** and CLIP[^1] and do the same with some short texts which are related to the image in various degrees. Then CLIP[^1] and **S-BRaCE** similarities are calculated. The table below reports these similarities, sorted according to the ranking received by CLIP[^1]

| Text                       | CLIP[^1] similarity                         | S-BRaCE similarity                          |
| -------------------------- | ------------------------------------------- | ------------------------------------------- |
| "dog with blue background" | <span style="color: #b6f0c0;">0.3244</span> | <span style="color: #b6f0c0;">0.2000</span> |
| "a dog"                    | <span style="color: #c2f0b6;">0.2709</span> | <span style="color: #d3f0b6;">0.1111</span> |
| "cute"                     | <span style="color: #d3f0b6;">0.2416</span> | <span style="color: #c2f0b6;">0.1154</span> |
| "blue"                     | <span style="color: #f0d8b6;">0.2389</span> | <span style="color: #f0c8b6;">0.0455</span> |
| "cat"                      | <span style="color: #f0c8b6;">0.2090</span> | <span style="color: #f0b6b6;">0.0000</span> |
| "house"                    | <span style="color: #f0b6b6;">0.1976</span> | <span style="color: #f0b6b6;">0.0000</span> |

S-BRaCE assigns the second highest similarity to "cute". Further, zero similarity is assigned to "cat" and "house". In general the agreement in ranking between CLIP and **S-BRaCE** will not be great. A simple reason why i believe this is, that **S-BRaCE** is less able to differentiate dissimilarities - many very dissimilar pairs will just result in zeros and not in gradual low scores. This is one limitation that is built-in in the setting itself. More on ranking agreement in the retrieval section further below.

#### Sparsity and Collisions

We now investigate the binary representations a bit closer - for now only on the validation dataset. A similar investigation on an unrelated dataset is planned.

##### Sparsity

Since the method is called "Sparse Binary Representation of Contrastive Embeddings" we should check if the representations are really sparse. We calculate the number of active
entries (active being: 1) for all text and image samples in the validation dataset and plot them:

<img src="/assets/img/plots/sparsenes.png" alt="alt drawing" width="500">

The median number of active entries for images is $10$ and for text $9$. The 97% quartile for them is $22$ and $20$ respectively. Given that the embedding dimension is $256$, this means that 97% of texts and images use not more than around 8% of the possible entries. As far as I am aware of there is no strict definition what constitutes a sparse vector, but this seems pretty sparse to me. To get a sense of how these ones are distributed we plot the binary representations for 500 images

<img src="/assets/img/plots/active_ent.png" alt="alt drawing" width="500">

This plot already provides a good impression on how well the entries are distributed. It is exactly what we want to see: there are not many entries active per row, and over the rows the active entries are well distributed. I.e. there is not one column which is massively over represented. This would hint at a representation collapse and uninformative binary representations. Stated differently: this image hints towards high mutual information between input images and output representation - exactly what we which for. In order to make sure no entry is actually overrepresented, we plot the normalized frequency being active per entry:

<img src="/assets/img/plots/norm_freq.png" alt="alt drawing" width="500">

The most frequent entries are active around 10%. My guess is that in a better **S-BRaCE** model this number would decrease and the activation should be even more uniform.

##### Collisions

The next question is how likely collisions occur. My definition of a collision is:
A collision has happened, when two different inputs of the same mode create the exact same binary representation. Let's check how many collisions occur in the validation dataset:
TBA

#### Zero-Shot Results

One way to check if a contrastive learning method has learned useful features is to use the resulting embeddings for Zero-Shot classification. We can do the same with **S-BRaCE**. We will start with [cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)[^11]. First, we embed the class names via **S-BRaCE** and plot the number of active entries for each class:
<img src="/assets/img/plots/cifar10_bin_txt.png" alt="alt drawing" width="500">

Quite surprisingly plane, car, cat, dog, horse are represented exactly by one entry. Naturally this reflects the typically large number of occurrences of these classes in datasets. Still, given the small capacity of 256 entries it is surprising that some of them seem to be completely reserved for one exact word/concept. Further, this raises the question if this might lead to many false positives. And indeed, the Zero-Shot performance is sobering. While CLIP[^1] reports over 90% accuracy, **S-BRaCE** only reaches 57%. Below the confusion matrix is depicted: 

<img src="/assets/img/plots/confusion.png" alt="alt drawing" width="500">

TBA: interpretation

#### Retrieval from a (somewhat) large dataset

We test how well images are retrieved via prompts from larger batches of data. The current experiments are preliminary. For this test we embed the Flickr30k[^8] dataset. This is, of corse not really considered large - modern similarity search methods like [faiss](https://github.com/facebookresearch/faiss) [^9] [^10] are able to handle billions of embeddings. Compared to this, the experiments here are comically small. Still, we get a taste on how well **S-BRaCE** is able to retrieve from a larger set of images based on a prompt. Again, currently we only provide qualitative results. The method is as follows: I sample a random text prompt form the Flickr30k[^8] and then plot the $k=5$ top retrievals for CLIP and **S-BRaCE** and plot them against each other. On top I plot the image that was associated with the prompt in the dataset.

| Target: A group of people hike in skis. | Result |
| :-------------------------------------: ||:------------------------------: |
![Image 1](/assets/img/plots/hikers.jpg) | ![Image 2](/assets/img/plots/hikers_result.png)

| Target: A man and a woman walk across an intersection on a sunny day. | Result |
| :-------------------------------------------------------------------: ||:------------------------------: |
![Image 1](/assets/img/plots/man_woman.jpg) | ![Image 2](/assets/img/plots/man_woman_result.png)

| Target: A brown dog runs through the water in the ocean. | Result |
| :------------------------------------------------------: ||:------------------------------: |
![Image 1](/assets/img/plots/dog.jpg) | ![Image 2](/assets/img/plots/dog_result.png)

While CLIP retrievals are in general better (twice in these three examples the target is contained in the results), **S-BRaCE** results do make sense. Training on more data will likely improve the retrieval results considerably.

## Acknowledgements

Many thanks to Rahul Siripurapu for reviewing the blog post and providing valuable feedback!

## Appendix

### Some additional Settings

It is still true that no clean hyper parameter search has been conducted, but: i did try out some additional settings. As with the reported experiments above, these currently are more anecdotal. Clean experiments will follow.

- **S-BRaCE** only with one linear layer: only one epoch is needed to train the model to convergences to a similar loss then with more layers. But the performance besides the training and validation loss was much worse then with one hidden layer.
- **S-BRaCE** with multiple layers (in contrast to one): the loss converges at a slightly higher level - however not only the validation loss but also the training loss levels out quickly. The performance is comparable to one hidden layer (reported above)

### Examples of $S(K)$ with a Minimum as Extremum

When first deriving the method, I believed that it might be enough to find a $K$ such that $\mathcal{S}(K) > \mathcal{S}(K+1)$ (except for the case $K^\ast = D$). For some readers it might be obvious that this is not the case - it took some time for me. In the end i tried to find counter examples. The following counterexample is a $\hat{\mathbf{v}}$ such that $\mathcal{S}(K; \hat{\mathbf{v}})$ has a minimum as extremum.

We construct a $\hat{\mathbf{v}}$ such that for a $\alpha \in [0, 1]$ we have that $(100 \cdot \alpha) \%$ of the mass is distributed to the first entry and $(100 \cdot [1-\alpha]) \%$ is distributed uniformly to the rest of the entries.
We set $\alpha = 0.5$ (TODO: get the exact interval where this works) and construct $\hat{\mathbf{v}}$:

$$
    \label{eq:counter-example-1}
    \sum_{i=1}^D \hat{\mathbf{v}}_i^2 = 1 \iff \hat{\mathbf{v}}_1^2 = 1 - \sum_{i=2}^D \hat{\mathbf{v}}_i^2
    %\implies \hat{\mathbf{v}}_1 = \sqrt{1 - \sum_{i=2}^D \hat{\mathbf{v}}_i^2}
$$

and since for all $ i, j \geq 2$ our entries are equal $\hat{\mathbf{v}}_i = \hat{\mathbf{v}}_j$, we can calculate their value as

$$
    \label{eq:counter-example-2}
    \hat{\mathbf{v}}_1^2 = 1 - \sum_{i=2}^D \hat{\mathbf{v}}_i^2 = 1 - (D-1) \hat{\mathbf{v}}_2^2 \implies
$$

$$
    \label{eq:counter-example-3}
    \sqrt{\frac{1 - \hat{\mathbf{v}}_1^2}{D-1}} = \hat{\mathbf{v}}_2
$$

where $\hat{\mathbf{v}}_2$ represents all $\hat{\mathbf{v}}_i$ for $i \geq 2$.
In order to distribute $50 \%$ of the mass to the first entry and equally to the rest, we have to set $\hat{\mathbf{v}}_1^2 = 0.5$ and $\hat{\mathbf{v}}\_2^2 = 0.00196078431372549$ which results in $\hat{\mathbf{v}}_1 = 0.7071067811865476$ and $\hat{\mathbf{v}}\_2 =  0.04428074427700476$. If we plot $S(K;\hat{\mathbf{v}})$ we get the following plot which clearly serves as a counter example to my initial assumption:

<img src="/assets/img/plots/counterex.png" alt="alt drawing" width="500">

Unfortunately this means that we need to check for the argmax until $K = D$.

## Citation Information

If you find **S-BRaCE** useful and intend to use it, please cite this blog via:

```bibtex
@misc{Ramsauer2024,
  author = {Hubert Ramsauer},
  title = {S-[TB]RaCE: Learning Sparse Binary Representation of Contrastive Embeddings},
  year = {2024},
  url = {https://myblog.com/your-post-url},
  note = {Accessed: YYYY-MM-DD}
}
```

## References

[^1]: A. Radford et al., _Learning Transferable Visual Models From Natural Language Supervision_. International Conference on Machine Learning, 2021
[^2]: A. Fürst et al., _CLOOB: Modern Hopfield Networks with InfoLOOB Outperform CLIP_. Neural Information Processing Systems, 2022
[^3]: X. Zhai et al., _Sigmoid Loss for Language Image Pre-Training_. International Conference on Computer Vision, 2023
[^4]: P. Sharma et al., _Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning_, Proceedings of ACL, 2018
[^5]: Dan Hendrycks, Kevin Gimpel, _Gaussian Error Linear Units (GELUs)_, arXiv preprint arXiv:1606.08415, 2016
[^6]: Ilya Loshchilov, Frank Hutter, _Decoupled Weight Decay Regularization_, InternationalConference on Learning Representations, 2019
[^7]: Diederik P. Kingma, Jimmy Ba, _Adam: A Method for Stochastic Optimization_, InternationalConference on Learning Representations, 2015
[^8]: P. Young et al., _From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions_, Transactions of the Association for Computational Linguistics, 2014
[^9]: M. Douze et al., _The Faiss library_, arXiv, 2024
[^10]: J. Johnson et al., _Billion-scale similarity search with GPUs_, IEEE Transactions on Big Data, Vol. 7, Page 535-547, 2019
[^11]: Alex Krizhevsky, _Learning multiple layers of features from tiny images_, Technical Report, 2009

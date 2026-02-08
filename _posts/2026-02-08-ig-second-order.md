---
title: "What Input×Gradient Misses: The Curvature Behind Integrated Gradients"
author: hubi
date: 2026-02-08 09:00:00 +0200
# description: Integrated Gradients is Input×Gradient with curvature correction
categories: [Blogging, ML]
tags: [integrated gradients, attribution, curvature, hessian]
math: true
pin: false
---

## Intro

Series: **Integrated Gradients** (Part 1 of ?)  
Nav: [Part 1]({% post_url 2026-02-08-ig-second-order %})

> This blog post assumes familiarity with Integrated Gradients[^1] and won't introduce the method

This is Part 1 of a short series on **Integrated Gradients** (IG). It is based on work I have been doing a very long time ago, not long after the paper, [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365)[^1], was published.

My main interest at that time was to use IG for optimization and/or pruning. This led me down a rabbit hole with some results that are at least not completely uninteresting (I hope).

The first result, presented in this post, might be obvious to the mathematically more inclined reader. In short it states that IG is exactly a first-order term corrected by a **path-averaged curvature** term. This first-order term is exactly **Input×Gradient**, introduced in [Not Just a Black Box](https://arxiv.org/abs/1605.01713)[^3]. This will be derived in this blog post. The original derivation was pretty messy. This blog presents a cleaned up version via integration by parts, which was absolutely not the path I initially took. In any case let's set up the notation.

---

## IG Formulation

**Notation.** Let $f: \mathbb{R}^n \rightarrow \mathbb{R}$ be a $C^2$ function. We write $\nabla f \in \mathbb{R}^n$ for the gradient and $\nabla^2 f \in \mathbb{R}^{n \times n}$ for the Hessian (we'll need this later). For vectors $a, b$, we use $a \odot b$ for elementwise multiplication.

Choose a baseline $x_0$ and define the displacement $s := (x - x_0)$. The straight path is

$$
\gamma(\alpha) = x_0 + \alpha s, \quad \alpha \in [0,1].
$$

Writing $\hat{x} := \gamma(\alpha)$, the IG attribution vector is

$$
\mathrm{IG}(x) = s \odot \int_0^1 \nabla f(\hat{x}) \, d\alpha.
$$

## A second order view

We first note that conveniently due to the linear path the path tangent is constant:

$$
\frac{d\gamma}{d\alpha} = s.
$$

By the chain rule, the gradient derivative along this path is

$$
\frac{d}{d\alpha} \nabla f(\gamma(\alpha))
= \nabla^2 f(\gamma(\alpha)) \frac{d\gamma}{d\alpha}
= \nabla^2 f(\hat{x}) s.
$$

<!-- Define the path-averaged gradient -->
<!---->
<!-- $$ -->
<!-- g_{\mathrm{IG}} := \int_0^1 \nabla f(\hat{x}) \, d\alpha. -->
<!-- $$ -->

Apply $\int u \, dv = uv - \int v \, du$ with

- $u = \nabla f(\hat{x}) \Rightarrow du = \bigl(\nabla^2 f(\hat{x}) s\bigr) d\alpha$
- $dv = d\alpha \Rightarrow v = \alpha$

to get

$$
\int_0^1 \nabla f(\hat{x}) \, d\alpha
= \Bigl[\alpha \nabla f(\hat{x})\Bigr]_0^1
- \int_0^1 \alpha \, \nabla^2 f(\hat{x}) s \, d\alpha.
$$

The boundary term is
$
[\alpha \nabla f(\hat{x})]_0^1 = \nabla f(x),
$
since $\hat{x}=x$ at $\alpha=1$ and $\hat{x}=x_0$ at $\alpha=0$. Therefore,

$$
\int_0^1 \nabla f(\hat{x}) \, d\alpha
 = \nabla f(x) - \underbrace{\int_0^1 \alpha \, \nabla^2 f(\hat{x}) \, d\alpha}_{\bar{H}} s.
$$

<!-- Rearranging yields the identity -->
<!---->
<!-- $$ -->
<!-- \nabla f(x) - g_{\mathrm{IG}} = \bar{H} s. -->
<!-- $$ -->

Multiplying both sides by $s$ elementwise gives the second-order form of IG:

$$
\mathrm{IG}(x) = s \odot \nabla f(x) - s \odot [\bar{H} s].
$$

<!-- So IG is a first-order term corrected by a path-averaged curvature term. This is an **exact decomposition**; no Taylor truncation is used. The identity shows that IG implicitly accounts for how the gradient changes between baseline and input. The baseline matters because it determines which curvature gets integrated. Because of the $\alpha$ weight, $\bar{H}$ emphasizes curvature closer to $x$ (near $\alpha=1$) more than curvature near the baseline. -->

### Connection to Input×Gradient

The first term, $s \odot \nabla f(x)$, is the **Gradient×Input-Difference** attribution. Many readers reserve **Input×Gradient** for the special case $x_0 = 0$, so that $s = x$; this is the Gradient×Input method discussed in [Not Just a Black Box](https://arxiv.org/abs/1605.01713)[^3], and summarized in [Ancona et al.](https://arxiv.org/abs/1711.06104)[^2]. IG adds the second-order correction $-s \odot [\bar{H} s]$. In particular, if the baseline is zero ($x_0 = 0$), then $s = x$ and

$$
\mathrm{IG}(x) = x \odot \nabla f(x) - x \odot [\bar{H} x],
$$

### 1D Example

Let $f(x) = x^2$ with baseline $x_0 = 0$. Then $s = x$, $\nabla f(x) = 2x$, and $\nabla^2 f = 2$.

![1D quadratic: IG vs Input×Gradient](/assets/img/posts/ig-series/ig-quadratic-1d.svg)

Integrated Gradients:

$$
\mathrm{IG}(x) = x \int_0^1 2\alpha x \, d\alpha = x \cdot \left[ x \alpha^2 \right]_0^1 = x \cdot x = x^2.
$$

Input×Gradient:

$$
x \odot \nabla f(x) = 2x^2.
$$

Curvature correction:

$$
\bar{H} = \int_0^1 \alpha \cdot 2 \, d\alpha = 1,
\quad -x \odot (\bar{H} x) = -x^2.
$$

As we can see, $\mathrm{IG}(x) = \text{Input×Gradient} + \text{Curvature correction} = 2x^2 - x^2 = x^2$. For a convex quadratic, Input×Gradient overshoots and the curvature term fixes it.

---

## Citation Information

If you find this useful, please cite this blog via:

```bibtex
@misc{Ramsauer2026,
  author = {Hubert Ramsauer},
  title = {What Input×Gradient Misses: The Curvature Behind Integrated Gradients},
  year = {2026},
  url = {https://HubiRa.github.io/posts/ig-second-order/},
}
```

## References

[^1]: M. Sundararajan, A. Taly, Q. Yan, "Axiomatic Attribution for Deep Networks", ICML 2017.

[^2]: M. Ancona, E. Ceolini, C. Öztireli, M. Gross, "Towards Better Understanding of Gradient-Based Attribution Methods for Deep Neural Networks", ICLR 2018.

[^3]: A. Shrikumar, P. Greenside, A. Shcherbina, A. Kundaje, "Not Just a Black Box: Learning Important Features Through Propagating Activation Differences", CoRR abs/1605.01713, 2016.

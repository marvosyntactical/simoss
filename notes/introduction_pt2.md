# Introduction Part 2

## Acceptance/Rejection Method

Unnormalized positive target density: $f:\mathbb{R}\to[0,-\inf], f\in L^1$

aim: simulate random numbers from $X\sim\hat{f}=\frac{f}{||f||\_1}$

Sketch: See introductory slides


### Algorithm For Acceptance-Rejection-Method

```python3
def u = z, y = 0
while u > f(y)/g(y9:
    generate u = Uniform([0,1])
    generate y  = inversion(g, u)
set x = y
```
* each loop is bernoulli-exp with $p=\frac{||f||\_1}{||g||\_1}$
* therefore loop length is geometrically distributed: $\mathbb{P}(N=k)(1-p)^{k-1}\cdot p$
* therefore expected loop time is $\mathbb{E}[N]=\frac{1}{p} = \frac{||g||\_1}{||f||\_1}$

## Example: Gaussian g for generation of Cauchy f

* $X\sim\mathcal{N}(0,1) \to$ pdf: $f(x)=\frac{\exp{\frac{-x^2}{2}}}{\sqrt{2\pi}}, x\in\mathbb{R}$
* Consider "Cauchy-like" distribution: $g(x)=\frac{1}{\sqrt{2\pi}}\frac{1}{1+\frac{x^2}{2}}$
* $f(x)=\frac{\exp{\frac{-x^2}{2}}}{\sqrt{2\pi}} = \frac{1}{\sqrt{2\pi}}\frac{1}{1+\frac{x^2}{2}+\ldots}$
* Generate $Z\sim g$ via _ITM_:
* $F(x)=\int_{-\inf}^xg(u)du = \frac{\sqrt{2}}{\sqrt{2}}\frac{1}{\sqrt{2\pi}}\int_{-\inf}^x\frac{1}{1 + (\frac{y}{\sqrt{2}})^2}du$


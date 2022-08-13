# introduction Part 3: Testing

## 1. Kolmogorov-Smirnov Test

Idea: Confirm Whether sampling distribution matches True CDF well

1. $X$ cont'ly distributed with cdf $F_X(t)=\mathbb{P}((X\leq t)$
2. Test: $X_1,\ldots,X_n$ are random numbers of $X$ ??
3. $\to$ compute the empirical cdf and compare to theoretical cdf!
4. \underline{empirical cdf}: $\hat{F}\_n(t) := \frac{1}{n}\sum_{i=1}^n\mathbb{1}\_{X_i\leq t}$,
5. define empirical cdf as random var: $X\_1,\ldots,X\n\sim\mathbb{P}\_X,\hat{X}=(X\_1,\ldots,X\_n)$
6. $\to$ is $F\_{\hat{X}}$ close to $F_X$ ?
7. define test stat: $K_n := \sqrt{n}\sup_{t\in\mathbb{R}}|F_{\hat{X}}(t) - F_X(t)|$
8. $K_n$ is independent of law $\mathbb{P}\_X$
9. For large $n$: $F_{K_n(t) (= \mathbb{P}(K_n\leq t) \approx 1-2\exp{-2t^2}$
10. $\to$ reject null hyp if $K_n$ has too large values
11. Select significance value $\alpha : \mathbb{P}(K_n\leq t_\alpha) = \alpha$
12. $\Rightarrow t_\alpha = \sqrt{\frac{1}{2}\log(\frac{2}{\alpha}}$ 
13. Reject $H_0$ at significance level $\alpha$ if $K_n\geq t_\alpha$

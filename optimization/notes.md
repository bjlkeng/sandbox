# 2024-09-12
* Ask Github CoPilot to generate Hessian of my function and it got it wrong!  Asked gpt-1o and it got it right!
* Using gpt-1o to help debug my Hessian (and compute it)!  I was able to give it the original numerical point, and it computed the gradient, hessian, inverse hessian and it matched my results!
* Next I asked it to do Newton's method on the original point and let's see what it gives me, and it matches!
* I just realized, Netwon's method is good at finding zero's of the gradient == local min OR max.  The algorithm was moving to find the maxima instead of minima!
* If the Hessian is not positive definite, it's not going to work!  That's why you can't use Newton's method straight up.  You need to ensure that it's a descent direction.

# 2024-09-06
* Last time added conjugate gradient descent
* Added a Nesterov, RMSProp, Adam, Hypergradient Descent
* CoPilot isn't bad, when I started to write the Adam function, it filled it in entirely right away!  Most likely because it appears on the web so often.
* However, when it tried to do it from the hypergradient (probably a lot less common), it just basically copied what I had for Nesterov momentum.  Very telling in what the LLM has seen before.

# 2024-08-29
* Using ChatGPT to gain intuition about positive definite, second order multivariate approximation, why Hessian is analagous to second derivative
* https://chatgpt.com/share/41a66828-1b06-485e-80b0-244d92528f0f
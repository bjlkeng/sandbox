2019-02-25

* Interesting: At the output of the mean pixels I add the `0.5` to `lambda x:
  4. * x + 0.5` and it converges so much faster!  Any value between (0.4-0.7)
  seems to work.  It must be the case that the extra constant helps "kickstart"
  the momentum because the bias is initalized to 0?  Not quite sure why it's so
  much faster.

2019-02-28
* Setup:
    * Just have simple biases hooked up to each m, s
    * Played around with activation function, range of activations and optimizer
* Did some debugging with Tensorboard:
    * A lot of the difference in convergence with changing the optimizer, SGD
      seems to converge fastest, probably because momentum isn't slowing things
      down
    * The "0.5" effect I think is probably because of the fact that biases start at 0.
    * It looks like my guess of having std. dev scaled to 0 to 7 is pretty good
      b/c 3.5 seems like a good "default".  3.5 translates to about 14 pixel
      std deviation ( 1 / exp(3.5) * pi / sqrt(3) -- logistic distribution)
    * There's some subtle interplay between m and s.  m seems to often get to the
      best values (in terms of distribution) but s seems to have a hard to
      optimizing for the ideal.

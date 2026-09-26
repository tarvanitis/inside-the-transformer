# The Mathematics You'll Need

Everything a transformer does reduces to a small set of operations on lists and grids of numbers.
This chapter builds that toolkit from the ground up (no prior mathematics assumed) and it is the one
place in the book where the core operations are defined in full. A handful of ideas that sit closer
to architecture than to arithmetic are taught where they are used. The end of this chapter maps them. Later chapters use these operations freely
and point back here rather than re-deriving them, so the chapter doubles as a reference you can
return to whenever a symbol stops making sense.

**In this chapter**

- **Linear algebra.** Vectors, matrices, matrix multiplication, dot products, cosine similarity,
  transpose, and rotation.
- **Probability.** Distributions, softmax, logits, temperature, top-p sampling, and the sigmoid.
- **Optimization.** Gradients, backpropagation, learning rate, and the Adam optimizer.
- **Functions and numerical methods.** The √d_k scaling factor, activation functions, and dropout.
- **Statistics.** Mean, variance and standard deviation, floating-point formats, and quantization.
- **Information.** Cross-entropy loss, perplexity, and KL divergence.

You do not need to read this chapter cover to cover before continuing. Skim it once, then use it as a
lookup. Every entry ends with **where it shows up** in the model, pointing to the chapter that puts
it to work.

## Linear algebra

### Vector (`v ∈ ℝⁿ`)

A *vector* is simply an ordered list of numbers, nothing more frightening than that. Picture a
shopping list, but instead of items you have numbers: `[0.91, 0.76, −0.14, 0.55]`. Each number is
called a *component* or *dimension*. (The heading `v ∈ ℝⁿ` is just shorthand: ℝ means "real numbers," ordinary numbers like 0.91 or −0.14. The small *n* says how many of them there are, and ∈ means
"is a member of." So it reads "v is a list of *n* ordinary numbers.") Inside a transformer, every
word is turned into a vector, a long list of numbers, usually a few thousand of them, that encodes
the word's meaning as a position in a high-dimensional space.

**Why it matters:** GPUs can only operate on numbers. The embedding table converts each word into a
vector so the model can do mathematics on language. Think of each vector as an address in "meaning
space": similar words live near one another.

**Worked example: "cat" as a tiny 4-dimensional vector.** Use only 4 dimensions to keep it simple.
After training, "cat" and "dog" might look like this:

```text
          dim1   dim2   dim3   dim4
"cat" = [ 0.91,  0.88,  0.02,  0.05 ]
"dog" = [ 0.89,  0.84,  0.03,  0.06 ]
```

Loosely, dim 1 ≈ animal-ness, dim 2 ≈ living-ness, dim 3 ≈ royal-ness, dim 4 ≈ past-tense-ness.

> **Notice:** "cat" and "dog" have very similar numbers in dims 1 and 2 (both are living animals),
> while dim 3 (royal-ness) is near zero for both. In a real model, all several-thousand dimensions
> work together and no single dimension carries a tidy label like this.

::: {.figure}
![](assets/figures/ch02/fig-vectors-in-space.svg)
:::

::: {.caption}
**Figure 2.1.** Words as points in space. Similar words (cat, dog) sit close together, while unrelated words (car, king) sit far apart.
:::

**Where it shows up:** token embeddings (Chapter 3); the query, key, and value vectors inside
attention (Chapter 5); and the feed-forward network, which transforms each token's vector
independently (Chapter 6).

### Matrix (`M ∈ ℝᵐˣⁿ`)

A *matrix* is a grid of numbers arranged in rows and columns, like a spreadsheet, or a times-table.
A matrix with 3 rows and 4 columns is a *3×4 matrix*. Every learned weight in a transformer (the
projection matrices, the feed-forward weights, the embedding table) is a matrix. Multiplying a
vector by a matrix *transforms* that vector: it rotates, stretches, or mixes its components.

**Why it matters:** All the "knowledge" a transformer picks up during training is stored in matrices.
When the model runs, it multiplies your token vectors by these matrices to update their
representations, layer by layer.

**Worked example: a 2×3 weight matrix.** Two rows, three columns, each cell a learned weight:

```text
      col1   col2   col3
W = [  0.5    0.2   −0.1 ]   ← row 1
    [  0.3    0.8    0.4 ]   ← row 2
```

This matrix transforms a 3-dimensional input vector into a 2-dimensional output vector. In a real
model, a query-projection matrix might turn a several-thousand-dimensional token into a
128-dimensional query: same idea, bigger numbers.

::: {.callout .plain}
Think of a matrix as a recipe. The columns are ingredients (input dimensions), the rows are dishes
(output dimensions), and each cell says how much of that ingredient goes into that dish.
:::

**Where it shows up:** the embedding matrix (Chapter 3); the query/key/value projections and the
output projection in attention (Chapter 5); the two feed-forward weight matrices (Chapter 6); and the
language-model head that turns the final vector into scores over the vocabulary (Chapter 8).

### Matrix multiplication (`C = A × B`)

When you multiply two matrices, each cell of the result is a *dot product* of a row from the first
matrix and a column from the second. This is the single most common computation in a transformer:
attention scores, feed-forward projections, and the final vocabulary scores all use it. Modern GPUs
are built specifically to do matrix multiplication as fast as physically possible.

**Why it matters:** Every linear transformation in a transformer is a matrix multiplication. The
attention formula is three of them stacked together:

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$

(Two symbols to note on first sight: the raised ᵀ / ⊤ means "transposed", the matrix flipped so its
rows become columns, covered under *Transpose* below, and √ means "square root.") The whole forward
pass is one long chain of these.

::: {.callout .plain}
Don't try to read that formula yet. It is here only to show you what the rest of the chapter is
building toward. Q, K and V are three versions of each token's vector, made by multiplying it by
three learned matrices: the *query* is what a token is looking for, the *key* is what it has to
offer, and the *value* is the content it hands over if it gets picked. d_k is simply how long those
vectors are, and softmax turns raw scores into percentages. Softmax is defined later in this
chapter, d_k under *Scaling factor*, and Chapter 5 assembles the whole thing properly.
:::

**Worked example: step by step.** Multiply a **2×3** matrix A by a **3×2** matrix B to get a **2×2**
result:

```text
     [ 1  2  3 ]        [  7   8 ]        [  58   64 ]
A =  [ 4  5  6 ]   B =  [  9  10 ]   C =  [ 139  154 ]
                       [ 11  12 ]
```

```text
C[0,0] = row 0 of A · col 0 of B = (1×7) + (2×9)  + (3×11) =  7 + 18 + 33 =  58
C[0,1] = (1×8) + (2×10) + (3×12) =  8 + 20 + 36 =  64
C[1,0] = (4×7) + (5×9)  + (6×11) = 28 + 45 + 66 = 139
C[1,1] = (4×8) + (5×10) + (6×12) = 32 + 50 + 72 = 154
```

> **Rule to remember:** A × B is only possible if A has as many columns as B has rows, and the result
> is shaped (A's rows) × (B's columns). In a transformer, a token vector `[1 × d_model]` times a
> projection `[d_model × d_k]` gives a query of shape `[1 × d_k]`.

::: {.figure}
![](assets/figures/ch02/fig-matmul-shapes.svg)
:::

::: {.caption}
**Figure 2.2.** Matrix multiplication. Each cell of the result is one row of A combined with one column of B. You can multiply two matrices only when the inner numbers match.
:::

**Where it shows up:** the query/key/value projections and both attention matmuls (Chapter 5); the
two feed-forward projections (Chapter 6); and the language-model head (Chapter 8).

### Dot product (`a · b`)

The dot product of two vectors multiplies each pair of matching numbers and sums the results into a
single number. That number measures *how much the two vectors point the same way*. Large and positive
means they are aligned (similar); near zero means they are perpendicular (unrelated); negative means
they point in opposite directions. Everyday picture: two people rate the same five films. Multiply
their scores film by film and add up the results. A big total means their tastes line up, a total
near zero means their preferences are unrelated.

**Why it matters:** The entire attention mechanism is built on dot products. A query dotted with a
key gives a similarity score, telling you how relevant one token is to another. The language-model head uses
them too: the final vector dotted with each word's embedding row gives that word's score.

**Worked example: step by step.**

```text
a = [3, 1, 4]   b = [2, 5, 1]
a · b = (3×2) + (1×5) + (4×1) = 6 + 5 + 4 = 15
```

Now two vectors that share nothing:

```text
c = [1, 0, 0]   d = [0, 1, 0]
c · d = (1×0) + (0×1) + (0×0) = 0     ← perpendicular, no similarity
```

> **In attention:** Take *The animal didn't cross the street because it was too tired.* To work out
> what "it" refers to, the model compares "it" against every other word. If the comparison score is
> 18 against "animal" but only 3 against "street," the model has learned that "it" refers to
> "animal." The dot product *is* the relevance score.

::: {.figure}
![](assets/figures/ch02/fig-dot-product-alignment.svg)
:::

::: {.caption}
**Figure 2.3.** The dot product measures how much two vectors point the same way. Pointing the same way gives a positive number, a right angle gives zero, and pointing apart gives a negative number.
:::

**Where it shows up:** attention scores (Chapter 5); the per-word scores produced by the
language-model head (Chapter 8).

### Cosine similarity (`cos(θ)`)

Cosine similarity measures the *angle* between two vectors, ignoring their length. It returns a value
between −1 and +1: vectors pointing the same way score 1 (identical direction), perpendicular vectors
score 0 (unrelated), and opposite vectors score −1. Everyday picture: it compares two compass needles
by the direction they point, not how long they are. "North vs north" scores 1, "north vs east" scores
0, "north vs south" scores −1.

**Why it matters:** It is the standard way to compare word embeddings. To find the words most similar
to "cat," you compute cosine similarity between the "cat" vector and every other word's vector. It is
also the foundation of *semantic search* and retrieval-augmented generation (Chapter 18).

The definition is the dot product divided by the two vectors' lengths:

$$\cos(\theta)=\frac{a\cdot b}{\lVert a\rVert\,\lVert b\rVert}$$

**Worked calculation.** Take the dot product of the two vectors, then divide by each vector's length.
A vector's length is √(sum of its squared components), where √ means "square root." Dividing by the
lengths cancels out size and leaves only direction:

```text
cat = [0.91, 0.88]   dog = [0.89, 0.84]
dot product = (0.91×0.89) + (0.88×0.84) = 0.810 + 0.739 = 1.549
|cat| = √(0.91² + 0.88²) = √(0.828 + 0.774) = √1.602 = 1.266
|dog| = √(0.89² + 0.84²) = √(0.792 + 0.706) = √1.498 = 1.224
cos(θ) = 1.5491 / (1.2659 × 1.2238) = 1.5491 / 1.5492 ≈ 0.9999   ← nearly identical
```

Those two toy vectors happen to point almost the same way, so the score sits at 0.9999. Real
embeddings are messier: genuinely related words like *cat* and *dog* usually land around 0.6–0.8, not
right next to 1. Typical values between real word embeddings look more like this (illustrative):

| Pair | Cosine similarity |
|------|-------------------|
| cat vs dog | 0.75 |
| cat vs king | 0.42 |
| cat vs car | 0.18 |

::: {.caption}
**Table 2.1.** Typical cosine similarities between real word embeddings (illustrative). The toy 2-D
vectors above are unusually close. Real related words sit lower, around 0.6–0.8.
:::

::: {.callout .lens}
**Geometric.** The dot product and cosine similarity are the same measurement seen two ways. The dot
product is "how much do these two arrows agree, scaled by how long they are". Cosine similarity
divides the lengths back out and leaves only the angle between them. Everything attention does with
similarity is one of these two, which is why so much of this book comes back to angles.
:::

::: {.figure}
![](assets/figures/ch02/fig-cosine-length-invariance.svg)
:::

::: {.caption}
**Figure 2.4.** Cosine similarity looks only at the angle, not the length. Vector b is twice as long as a but points the same way, so their cosine is 1 even though the dot product is larger.
:::

**Where it shows up:** measuring semantic similarity of embeddings (Chapter 3); scaled dot-product
attention is proportional to cosine similarity when the vectors are normalized (Chapter 5); and
semantic retrieval in RAG (Chapter 18).

### Transpose (`Aᵀ`)

Transposing a matrix flips it along its diagonal: rows become columns and columns become rows. If A
is 3 rows × 5 columns, then Aᵀ is 5 rows × 3 columns. This is needed constantly in transformer
mathematics, above all in the attention formula, where Q is multiplied by Kᵀ.

**Why it matters:** The attention formula requires `Q × Kᵀ`. Q has shape `[seq × d_k]` and so does K.
To multiply them, K must be transposed so the inner dimensions line up. Without the transpose, the
multiplication is undefined.

**Worked example.**

```text
       [ 1  2  3 ]               [ 1  4 ]
A =    [ 4  5  6 ]   (2×3)  →    [ 2  5 ]   (3×2) = Aᵀ
                                [ 3  6 ]
```

The rows of A have become the columns of Aᵀ.

> **In attention:** Q has shape `[5 tokens × 64 dims]` and so does K. Transposing K to `[64 × 5]`
> makes `Q × Kᵀ` a `[5 × 5]` score matrix, one score for every pair of tokens.

::: {.figure}
![](assets/figures/ch02/fig-transpose-diagonal.svg)
:::

::: {.caption}
**Figure 2.5.** Transposing a matrix flips it across its diagonal, so each row becomes a column. The first row of A becomes the first column of Aᵀ.
:::

**Where it shows up:** the `Q × Kᵀ` score computation in attention (Chapter 5); and weight tying,
where the language-model head reuses the transposed embedding matrix (Chapter 8).

### Rotation, or RoPE (`R(θ, pos)`)

Rotation means spinning a vector by an angle within its space. In two dimensions, rotating `[1, 0]`
by 90° gives `[0, 1]`. Modern transformers use *rotary position embeddings* (RoPE), which encode a
token's position in the sequence by rotating its query and key vectors by an angle proportional to
that position: the token at position 1 is rotated a little, position 2 twice as much, and so on.

**Why it matters:** Attention has no built-in sense of word order. On its own it sees an unordered
bag of tokens. RoPE bakes position into the geometry. After rotating, the dot product between a query
at position *m* and a key at position *n* depends only on the *gap* `m − n`, never on where the pair
sits in the sequence. Across the many rotation frequencies RoPE uses, the dot product also tends to
fall off as the gap widens. Position becomes part of the vectors themselves rather than a separate
lookup table. (Positional encoding is covered in full in Chapter 4. This entry is just the
geometric primitive it rests on.)

**Intuition: rotating a 2D vector.** The rotation matrix is

$$R(\theta)=\begin{bmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{bmatrix}$$

Rotate `[1, 0]` by 45°:

```text
x' = cos(45°)×1 − sin(45°)×0 = 0.707
y' = sin(45°)×1 + cos(45°)×0 = 0.707
→ result: [0.707, 0.707]
```

::: {.callout .idea}
When the query and key of two nearby tokens are rotated by similar amounts, their dot product stays
high. As the tokens move apart, their rotations diverge and the dot product tends to fall, which is
how distance gets encoded.
:::

::: {.figure}
![](assets/figures/ch02/fig-rotation-rope.svg)
:::

::: {.caption}
**Figure 2.6.** Rotation turns a vector by an angle. RoPE uses this to record position: two tokens are rotated by their positions, so the angle between them depends only on how far apart they are.
:::

**Where it shows up:** positional encoding, applied by rotating Q and K inside the attention module
(Chapters 4 and 5).

## Probability

### Probability distribution (`P(x)`)

A probability distribution assigns a probability (a number between 0 and 1) to every possible
outcome, and all the probabilities must sum to exactly 1. In a transformer, the final output is a
probability distribution over the entire vocabulary: each token gets a probability of being the next
one. Sampling means drawing one token from this distribution.

**Why it matters:** The model doesn't deterministically pick one word. It expresses uncertainty as a
full distribution. That is what lets it be creative: different runs can sample different words. It
also lets us measure how confident the model is.

**Worked example: next-token distribution after "The cat sat on the".**

| Next token | Probability |
|------------|-------------|
| mat | 0.62 |
| floor | 0.18 |
| roof | 0.10 |
| table | 0.06 |
| other… | 0.04 |

::: {.caption}
**Table 2.2.** A next-token probability distribution over four candidate words.
:::

```text
All probabilities sum to 1.0:  0.62 + 0.18 + 0.10 + 0.06 + 0.04 = 1.00 ✓
```

::: {.callout .idea}
Even though "mat" has the highest probability (0.62), the model might still output "floor" or
"roof." This randomness is why an LLM's output varies between runs.
:::

**Where it shows up:** the output softmax layer (Chapter 8); sampling controls like temperature and
top-p (Chapter 8); and cross-entropy loss during training (Chapter 13).

### Softmax (`softmax(z)`)

Softmax takes a list of any real numbers (positive, negative, large, small) and converts them into
a valid probability distribution: every value becomes positive, and together they sum to 1. It raises
*e* (Euler's number, ≈ 2.718) to the power of each number, then divides each by the total. Bigger
input → much bigger output, so the largest number dominates. Everyday picture: it turns a set of raw
scores into slices of a pie that always add up to 100%, handing the biggest scores by far the biggest
slices.

$$\mathrm{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}$$

**Why it matters:** It appears in two critical places: (1) in attention, to turn raw dot-product
scores into weights that sum to 1; (2) at the output layer, to turn logit scores into next-token
probabilities. Without it you'd have raw numbers that don't behave like probabilities.

**Worked calculation: step by step.** Raw logit scores for three words: mat = 3.0, floor = 1.5,
table = 0.2.

```text
Step 1: raise e to the power of each logit
  e^3.0 = 20.09   e^1.5 = 4.48   e^0.2 = 1.22
Step 2: sum them
  total = 20.09 + 4.48 + 1.22 = 25.79
Step 3: divide each by the sum
  P(mat)   = 20.09 / 25.79 = 0.779   (77.9%)
  P(floor) =  4.48 / 25.79 = 0.174   (17.4%)
  P(table) =  1.22 / 25.79 = 0.047   ( 4.7%)
Sum: 0.779 + 0.174 + 0.047 = 1.000 ✓
```

::: {.callout .idea}
The exponential amplifies differences. Logits 3.0 vs 1.5 look "only" 2× apart, but after softmax the
probabilities are about 4.5× apart. A small logit lead becomes a big probability lead.
:::

::: {.figure}
![](assets/figures/ch02/fig-softmax-bars.svg)
:::

::: {.caption}
**Figure 2.7.** Softmax turns raw scores into probabilities that add up to 1, widening the gaps so the top score takes a larger share.
:::

**Where it shows up:** attention weights (Chapter 5); the output layer (Chapter 8); temperature
scaling before sampling (Chapter 8).

### Logits (`z ∈ ℝᵛ`)

Logits are the raw, unnormalized scores the language-model head produces, one number per vocabulary
token. They are not probabilities: they don't sum to 1 and can be any real number (negative, zero,
50, −10). Think of them as "votes" before counting. Softmax converts the votes into proper
probabilities. A logit of 8 for "cat" and 2 for "dog" means the model strongly favors "cat."

**Why it matters:** The model works with logits internally because it is more numerically stable and
efficient. You only need actual probabilities at the very last step (sampling). In training, the
cross-entropy loss can be computed directly from logits, with no need to form the softmax first.

**Worked example: logits vs probabilities.**

```text
Raw logit scores from the LM head (not probabilities!)
  "cat"   → logit =  8.2
  "dog"   → logit =  7.1
  "table" → logit =  1.4
  "car"   → logit = −2.1     ← negative is fine here

After softmax → proper probabilities
  "cat"   → 0.750   (75.0%)
  "dog"   → 0.249   (24.9%)
  "table" → 0.001   ( 0.1%)
  "car"   → 0.000   ( 0.0%)
```

**Where it shows up:** the language-model head (Chapter 8); the input to the output softmax and to
sampling (Chapter 8); and cross-entropy loss, which reads logits directly (Chapter 13).

### Temperature (`τ`)

Temperature is a number you divide the logits by before applying softmax. At τ = 1 (the default) the
distribution is unchanged. At τ < 1 (say 0.3) you enlarge the gaps between logits, so the distribution
becomes "sharper" and the top token dominates. At τ > 1 (say 1.5) you flatten the gaps, so
lower-probability tokens get more of a chance, and the output becomes more random and creative.

(The symbol is a Greek tau. This book writes temperature as τ rather than *T*, because *T* is already
the sequence length everywhere else, including in the model code of Chapters 10 and 11.)

**Why it matters:** It is a single dial for creativity vs consistency. Code generation typically uses
τ ≈ 0.2 (predictable output), and creative writing uses τ ≈ 0.8–1.0. τ = 0 means greedy decoding:
dividing by zero is undefined, so it is implemented as simply taking the most probable token (argmax).

**Worked example: same logits `[2.0, 1.0, 0.5]`, softmax after dividing by τ.**

```text
τ = 0.5 (sharper):  [0.84, 0.11, 0.04]   ← top token dominates
τ = 1.0 (default):  [0.63, 0.23, 0.14]
τ = 1.5 (flatter):  [0.53, 0.27, 0.20]   ← more random
```

::: {.figure}
![](assets/figures/ch02/fig-temperature.svg)
:::

::: {.caption}
**Figure 2.8.** Temperature reshapes the choice. A low temperature makes the top token more certain, and a high temperature spreads the chances out.
:::

**Where it shows up:** inference-time sampling, applied to the logits before softmax (Chapter 8).

### Top-p (nucleus) sampling (`p ∈ (0, 1]`)

Top-p sampling first sorts all vocabulary tokens by probability (highest first), then keeps adding
tokens to a candidate set until their probabilities sum to at least *p*. Only tokens in that set can
be sampled. At p = 0.9 you keep just enough tokens to cover 90% of the probability mass and discard
the rest, cutting off the unlikely "garbage" tail while keeping meaningful variety.

**Why it matters:** Pure temperature sampling can still, by chance, pick a very unlikely token.
Top-p is a safety net: whatever the temperature, the model can't select a token from the deep tail.

**Worked example: top-p = 0.90.**

```text
Sorted probabilities after softmax          (top-p = 0.90)
Token       Prob    Cumulative
"mat"       0.62    0.62   ← included
"floor"     0.18    0.80   ← included
"roof"      0.10    0.90   ← included (hits the threshold)
"table"     0.06    0.96   ← discarded (past the threshold)
"sofa"      0.02    0.98   ← discarded
"airplane"  0.001   …      ← discarded
→ Sample from {"mat", "floor", "roof"} only
```

> **Combined usage:** Most production systems use temperature and top-p together. Temperature shapes
> the distribution, then top-p clips the tail. Typical settings: τ = 0.7, p = 0.95.

**Where it shows up:** inference-time sampling, applied after softmax (Chapter 8).

### Sigmoid (`σ(x)`)

Sigmoid squashes a single number into the range 0 to 1. Unlike softmax, which works on a whole list
and makes it sum to 1, sigmoid works on one number at a time. Large positives approach 1, large
negatives approach 0, and 0 maps to exactly 0.5. Its graph is an S-curve.

$$\sigma(x)=\frac{1}{1+e^{-x}}$$

**Why it matters:** It appears in gating mechanisms and as part of an activation function. The SiLU
activation used in modern feed-forward layers is x · σ(x). Sigmoid also shows up in binary
classification heads.

**Worked examples.**

```text
σ( 0) = 1 / (1 + e⁰ ) = 1 / 2.000 = 0.500
σ( 2) = 1 / (1 + e⁻²) = 1 / 1.135 = 0.881
σ( 5) = 1 / (1 + e⁻⁵) = 1 / 1.007 = 0.993
σ(−2) = 1 / (1 + e² ) = 1 / 8.389 = 0.119
σ(−5) = 1 / (1 + e⁵ ) = 1 / 149.4 = 0.007
```

This σ is unrelated to the σ used for standard deviation later in this chapter. The overload is
standard notation, and the context always makes clear which is meant.

::: {.figure}
![](assets/figures/ch02/fig-sigmoid.svg)
:::

::: {.caption}
**Figure 2.9.** The sigmoid function squashes any number into a value between 0 and 1, crossing 0.5 at x = 0.
:::

**Where it shows up:** the SiLU activation inside the feed-forward network (Chapter 6).

## Optimization

### Gradient (`∇L`)

A gradient is a vector of partial derivatives, one per weight in the network. Each answers: "if I
nudge this weight up a little, does the loss rise or fall, and by how much?" The gradient points in
the direction that *increases* loss most steeply, so we move the weights the opposite way to reduce
it. Everyday picture: standing on a foggy hillside, you feel which way the ground slopes up most
steeply (that's the gradient) and step the other way to get lower. (The symbol ∇, "nabla", just
means "the gradient of," and *L* is the loss.)

**Why it matters:** With billions of parameters you can't try every combination. Gradients give the
exact direction to adjust each weight. Computing all of them at once is the job of backpropagation.

::: {.callout .plain}
A derivative is just a slope. Nudge *w* up by a hair and the derivative tells you how far *L* moves,
and in which direction. For L = (w − 3)² the slope works out to 2 × (w − 3), and you can check that
by hand: at w = 5, L is 4; at w = 5.01, L is 4.0401. That is a rise of 0.04 for a nudge of 0.01, a
slope of about 4, which is what 2 × (5 − 3) predicts. You will never have to work out a derivative
yourself, the framework does it. Knowing that it means "slope" is enough to follow everything ahead.
:::

**Worked example: one weight.** Minimize L = (w − 3)². With only one weight there is just one
number, so here the "vector" of partial derivatives is a single value.

```text
dL/dw = 2 × (w − 3)                  ← the gradient

At w = 5:  dL/dw = 2×(5−3) = +4       ← positive: move w down
At w = 1:  dL/dw = 2×(1−3) = −4       ← negative: move w up
At w = 3:  dL/dw = 2×(3−3) =  0       ← zero: w = 3 is the minimum

Update rule: w_new = w − 0.1 × gradient
Starting at w = 5:  5 → 4.6 → 4.28 → 4.02 → … → 3.0
```

Each step follows the weight-update rule, which recurs everywhere in training:

`w_new = w − α∇L`

::: {.figure}
![](assets/figures/ch02/fig-gradient-descent.svg)
:::

::: {.caption}
**Figure 2.10.** Gradient descent walks downhill to the lowest point of the loss. The step size (the learning rate) matters: too small and it crawls, too large and it overshoots and bounces.
:::

**Where it shows up:** training, where every weight is updated from its gradient (Part IV, Chapter 13).

### Backpropagation (`∂L/∂w`)

Backpropagation is the algorithm that computes the gradient for every weight in a deep network in two
passes. In the *forward pass*, the input flows through the network to produce an output and a loss.
In the *backward pass*, the loss signal flows backwards through each layer, applying the chain rule of
calculus to work out each weight's contribution. (The heading `∂L/∂w` reads "how much the loss L
changes when weight w changes a tiny bit", and the curly ∂ marks a partial derivative.)

**Why it matters:** A transformer may have tens of billions of weights. Computing each gradient
separately would take one forward pass per weight. Backprop gets them all in roughly the cost of two
forward passes, which is what makes training feasible at all.

**The chain rule: why it works.** Picture a three-layer network. Call the output of layer 1 `out₁`,
of layer 2 `out₂`, and of layer 3 `out₃`, the final prediction the loss is computed from. `w₁` is a
weight in layer 1. To see how that early weight affects the loss, multiply the local gradients along
the path from the loss back to it:

$$\frac{\partial L}{\partial w_1}=\frac{\partial L}{\partial \mathrm{out}_3}\cdot\frac{\partial \mathrm{out}_3}{\partial \mathrm{out}_2}\cdot\frac{\partial \mathrm{out}_2}{\partial \mathrm{out}_1}\cdot\frac{\partial \mathrm{out}_1}{\partial w_1}$$

The four numbers below are made up, but each is something the network can work out from its own
layer alone, without knowing anything about the others. That locality is the whole trick.

```text
Example with numbers (each term computed locally, then multiplied):
  ∂L/∂out₃   = 2.0     ← computed at the output
  ∂out₃/∂out₂ = 0.8    ← layer 3 local gradient
  ∂out₂/∂out₁ = 0.5    ← layer 2 local gradient
  ∂out₁/∂w₁   = 0.3    ← layer 1 local gradient
  ∂L/∂w₁ = 2.0 × 0.8 × 0.5 × 0.3 = 0.24
```

**Where it shows up:** every training step (Part IV, Chapter 13); residual connections exist partly
to give these gradients a clean path back through a deep stack (Chapter 6).

### Learning rate (`α`)

The learning rate is a small positive number (e.g. 0.0003) that sets how big each weight update is.
It multiplies the gradient to give the actual step. Too large and the weights overshoot the minimum
and training becomes unstable, too small and training crawls. Modern LLM training uses a *schedule*:
start small (warm-up), rise to a peak, then decay slowly.

**Why it matters:** It is the single most important hyperparameter. A wrong learning rate can wreck a
training run worth millions of GPU-hours. Getting it right (usually by experiment) is a core
engineering challenge.

**Worked example: the same weight, three learning rates** (gradient 4.0 at w = 5, minimum at w = 3).

```text
α = 0.01 (too small):  5 → 4.96 → 4.92 → … → 3.0 eventually, but very slowly
α = 0.10 (good):       5 → 4.6  → 4.28 → 4.02 → … → 3.0  (converges)
α = 1.00 (too large):  5 → 1.0  → 5.0  → 1.0 → … oscillates forever

Typical LLM schedule:
  steps     0 → 2000:  α increases linearly       (warm-up)
  steps  2000 → 1M:    α decays slowly            (cosine decay)
  end:      α ≈ 0.1 × peak learning rate
```

**Where it shows up:** every optimizer step during training (Part IV, Chapter 13).

### Adam optimizer (`Adam`)

Adam (Adaptive Moment Estimation) is the optimizer used to train virtually all modern LLMs. Instead
of updating each weight straight from the raw gradient, it keeps two running averages per weight: a
*momentum* term (the average of past gradients) and a *velocity* term (the average of past squared
gradients). These adapt the step size per weight: noisy gradients get smaller steps, consistent
ones get larger steps.

$$w_{\text{new}} = w - \alpha\,\frac{\hat m}{\sqrt{\hat v}+\epsilon}$$

**Why it matters:** Plain gradient descent treats every weight the same. Adam adapts the step to each
weight's history, converging faster and more stably, and its momentum helps it roll through flat
stretches and sharp turns in the loss landscape.

**Worked example: one step** (β₁ = 0.9, β₂ = 0.999, α = 0.001, gradient g = 0.4, step t = 1).

```text
Step 1: momentum       m = β₁·m_prev + (1−β₁)·g   = 0.9·0 + 0.1·0.4   = 0.04
Step 2: velocity       v = β₂·v_prev + (1−β₂)·g²   = 0.999·0 + 0.001·0.16 = 0.00016
Step 3: bias-correct   m̂ = m / (1 − β₁ᵗ) = 0.04 / 0.1   = 0.4
                       v̂ = v / (1 − β₂ᵗ) = 0.00016 / 0.001 = 0.16
Step 4: update         w_new = w − α · m̂ / (√v̂ + ε)
                             = w − 0.001 · 0.4 / (0.4 + 1e-8)
                             = w − 0.001     ← a controlled step
```

**Where it shows up:** the optimizer for almost all LLM training (Part IV, Chapter 13).

## Functions and numerical methods

### Scaling factor √d_k (`1/√d_k`)

In the attention formula, the raw dot products `Q·Kᵀ` are divided by √d_k (the square root of the
query/key dimension) before softmax. This keeps the scores from growing too large in high dimensions.
Without it, softmax would output values very close to 0 or 1, the gradients would vanish, and the
model would stop learning.

$$\mathrm{scores}=\frac{Q K^{\top}}{\sqrt{d_k}}$$

**Why it matters:** In d_k dimensions, the typical size of a dot product grows in proportion to √d_k.
The reason is short: if each component has variance 1, the dot product of two independent vectors
sums d_k such products, so its variance is d_k and its standard deviation is √d_k. Dividing by √d_k
cancels that growth and keeps the scores in a range where softmax gives useful gradients.

**Worked example: why large scores hurt** (d_k = 64, so divide by √64 = 8).

```text
Without scaling (scores get large):
  scores  = [24.1, 0.8, 0.3]
  softmax = [~1.000, ~0.000, ~0.000]   ← collapsed! one token takes all the weight
  gradient ≈ 0 everywhere              ← the model can't learn

With scaling (÷ 8):
  scores  = [24.1/8, 0.8/8, 0.3/8] = [3.01, 0.10, 0.04]
  softmax = [0.90, 0.05, 0.05]         ← spread out, useful gradients
```

**Where it shows up:** scaled dot-product attention (Chapter 5).

### Activation functions (ReLU / SiLU / GELU)

Activation functions add non-linearity to the feed-forward network. Without them, stacking layers
would collapse to a single linear transformation, useless for learning complex patterns. **ReLU**
outputs 0 for negatives and x for positives (simple and fast). **SiLU** = x · σ(x) is smoother and
lets small negatives through. It is common in modern transformers. **GELU** = x · Φ(x) is a similar
smooth curve used in BERT and GPT, where Φ(x) is the probability that a standard bell-curve value
falls below x. In practice a fast tanh-based approximation of Φ is used.

`SiLU(x) = x · σ(x)` and `GELU(x) = x · Φ(x)`

**Why it matters:** Non-linearity is what lets deep networks approximate arbitrary functions, not
just linear ones. Without it, 96 transformer layers would collapse into one. The feed-forward
network's expand-then-compress design only works because a non-linearity sits between the two
projections.

**Worked examples: same inputs, three activations.**

```text
x:        −2.0   −0.5    0.0    0.5    2.0
ReLU(x):   0.00   0.00   0.00   0.50   2.00
SiLU(x):  −0.24  −0.19   0.00   0.31   1.76
GELU(x):  −0.05  −0.15   0.00   0.35   1.95
```

::: {.callout .deepdive}
**Gated FFNs (SwiGLU).** Most modern LLMs (Llama, PaLM, Mistral, Qwen) replace the plain
`W₂ · SiLU(W₁·x)` feed-forward with a *gated* variant, SwiGLU, that adds a third weight matrix:

`FFN(x) = W₂(SiLU(W₁x) ⊙ W₃x)`

where ⊙ is elementwise multiplication. The extra gate branch `(W₃·x)` lets the network modulate its
own activations. To keep the parameter count fixed, the hidden width is scaled to ~⅔·(4·d_model).
GeGLU is the same idea with GELU.
:::

**Where it shows up:** the feed-forward network in every transformer block (Chapter 6).

### Dropout (`p_drop`)

During training only, each activation is randomly set to zero with probability p (typically 0.1–0.2),
and the survivors are scaled up by 1/(1−p) to compensate. This stops the network relying on any single
neuron, which improves generalization. At inference, dropout is switched off entirely. Everyday
picture: a sports team runs drills with a few random players benched each session, so everyone learns
to cover for the others and the whole team is more resilient on game day.

**Why it matters:** Large models can memorize training data instead of learning general patterns
(overfitting). Dropout is a regularizer: by disabling random neurons it forces redundant
representations that hold up better on new inputs.

**Worked example: p = 0.5.**

```text
Original activations:  [0.8, 1.2, 0.4, 0.9, 1.5, 0.3]
Random mask (1=keep):  [1,   0,   1,   0,   1,   1  ]
After dropout:         [0.8, 0.0, 0.4, 0.0, 1.5, 0.3]
Scale survivors ×1/(1−0.5)=2:  [1.6, 0.0, 0.8, 0.0, 3.0, 0.6]
```

**Where it shows up:** Chapters 10 and 11, where it appears in the PyTorch model code, and switched off
at inference. Note that large-scale pre-training of modern LLMs often sets dropout to 0, so you will
not see it in the Llama-style configurations later in the book.

## Statistics

### Mean (`μ`)

The mean (average) is the sum of the values divided by how many there are. It marks the "center" of a
set of numbers. In a transformer it is the first step of layer normalization: the mean is subtracted
from a token's activations to center them around zero.

$$\mu=\frac{1}{n}\sum_{i} x_i$$

**Why it matters:** Centering around zero removes systematic bias from activations. If every value is
shifted by a constant, subtracting the mean removes the shift, so layers don't accumulate drift.

**Worked calculation.**

```text
x = [2.0, 4.0, 6.0, 8.0]
μ = (2 + 4 + 6 + 8) / 4 = 20 / 4 = 5.0
Centered: x − μ = [−3.0, −1.0, +1.0, +3.0]   (their average is now 0 ✓)
```

**Where it shows up:** step one of layer normalization (Chapter 6).

### Variance and standard deviation (`σ², σ`)

Variance measures how spread out numbers are around their mean. Standard deviation is its square root
and shares the units of the data. A small standard deviation means the values cluster tightly, and a
large one means they spread wide. In layer normalization we divide by the standard deviation to
rescale the spread to ≈ 1.

$$\sigma^2=\frac{1}{n}\sum_{i}(x_i-\mu)^2 \qquad \sigma=\sqrt{\sigma^2}$$

**Why it matters:** After centering, the scale still needs standardizing. If activations have a
standard deviation of 1000, centring alone doesn't help, because the values are still huge. Dividing by σ
rescales everything to unit spread, keeping training stable.

**Worked calculation** (using the centered vector from *Mean*).

```text
Centered: [−3.0, −1.0, +1.0, +3.0]
σ² = ((−3)² + (−1)² + (+1)² + (+3)²) / 4 = (9 + 1 + 1 + 9) / 4 = 5.0
σ  = √5.0 = 2.236
Normalized: (x − μ) / σ = [−1.342, −0.447, +0.447, +1.342]   (mean 0, std 1 ✓)
```

**Where it shows up:** step two of layer normalization (Chapter 6).

### Floating point (`f16 / bf16`)

Computers store numbers in a fixed number of bits. **float32** (4 bytes) is high precision with a
safe range. **float16** (2 bytes) halves the memory but overflows for very large numbers and underflows to zero
for very small ones, which is why gradients, often tiny, are where it breaks first.
**bfloat16** (2 bytes) keeps float32's range but drops precision, and it is the usual choice for training large
models. Inference often uses float16 or even int8/int4 (quantization).

**Why it matters:** Memory is the main bottleneck for large-model inference. Precision directly sets
how much fits on a GPU.

**Worked example: a 7B-parameter model.**

```text
7,000,000,000 params × bytes per param:
  float32 (4 bytes): 28.0 GB   ← weights alone need a 40GB data-center GPU
  float16 (2 bytes): 14.0 GB   ← weights alone fit a 16GB GPU
  int8    (1 byte):   7.0 GB   ← weights alone fit a consumer GPU
  int4    (0.5 byte): 3.5 GB   ← weights alone fit a high-end laptop

Those are weights only. Add roughly the same again for the KV cache and activations, so budget
about 2× the weight size in VRAM to actually run the model.
```

**Where it shows up:** memory and precision throughout inference and training (Chapters 8 and 20).

### Quantization (`Q(w)`)

Quantization compresses weights from 16- or 32-bit floats to lower-precision integers (int8 = 1 byte,
int4 = 0.5 byte). Each weight is approximated by the nearest quantized value. Quality dips slightly
but is often acceptable, and it's what lets tools like llama.cpp run large models on consumer
hardware.

**Why it matters:** It makes large models runnable without data-center GPUs. A 70B model needs 140 GB
in float16 but about 35 GB at int4, which is a single high-end workstation GPU.

**Worked example: int8 quantization of 4 weights.**

```text
Original float32 weights: w = [0.312, −0.891, 0.544, −0.127]

Step 1: scale factor
  max_abs = 0.891
  scale   = 0.891 / 127 = 0.007016       (int8 range: −128…127)
Step 2: quantize (divide by scale, round)
   0.312 / 0.007016 =  44.47 →  44
  −0.891 / 0.007016 = −127.0 → −127
   0.544 / 0.007016 =  77.54 →  78
  −0.127 / 0.007016 = −18.10 → −18
Step 3: dequantize (multiply back)
   44 × 0.007016 =  0.309   (was  0.312, error 0.003)
  −127 × 0.007016 = −0.891  (was −0.891, error 0.000)
   78 × 0.007016 =  0.547   (was  0.544, error 0.003)
  −18 × 0.007016 = −0.126   (was −0.127, error 0.001)
```

::: {.callout .caution}
**Quality trade-off.** Errors are tiny per weight, but with billions of weights they add up.
4-bit loses more quality than 8-bit. Models are sometimes fine-tuned after quantization to recover.
:::

::: {.figure}
![](assets/figures/ch02/fig-quantization.svg)
:::

::: {.caption}
**Figure 2.11.** Quantization rounds each weight to the nearest value on a coarse grid, trading a little accuracy for much less memory.
:::

**Where it shows up:** on-device and efficient inference (Chapters 8 and 20).

## Information

### Cross-entropy loss (`L = −log P(y)`)

Cross-entropy loss is the single training objective for language models. At each position, the model
predicts a probability distribution over the vocabulary. The loss is minus the log-probability it
assigned to the *correct* next token. Assign 90% to the right token and the loss is low
(−log 0.9 = 0.105). Assign 1% and it is high (−log 0.01 = 4.605). Training minimizes the average of
this loss over billions of tokens.

**Why it matters:** This measures "how surprised was the model by the correct answer?" Minimizing
surprise across the training data forces it to learn grammar, facts, and reasoning: everything that
makes natural language less surprising.

**Worked examples: feel the scale.**

```text
L = −log(P(correct token))
  P = 0.90 →  L = 0.105    ← confident and right
  P = 0.50 →  L = 0.693    ← moderate
  P = 0.10 →  L = 2.303    ← confused
  P = 0.01 →  L = 4.605    ← very wrong: big gradient, big update

GPT-2 training loss ran ~5.0 → ~3.0. Modern frontier models reach ~1.5–2.0.
```

> **Why the logarithm?** It turns multiplied probabilities into added losses (cleaner math) and
> punishes confident-but-wrong predictions hard. P = 0.001 for the correct token gives a loss of
> 6.9, a very strong corrective signal.

**Where it shows up:** the training objective, computed at every token position (Part IV, Chapter 13).

### Perplexity (`PPL`)

Perplexity is the exponential of the average cross-entropy loss. It answers "on average, how many
equally likely options did the model feel it was choosing between at each token?" A perplexity of 10
means it was about as unsure as picking from 10 equally likely words. A perplexity of 2 means it was
usually down to 2. Lower is better, and it is the standard metric for comparing language models.

`PPL = e^L̄`, where L̄ is the average cross-entropy loss.

**Why it matters:** Loss values like 2.3 are hard to interpret. Perplexity turns them into a
"branching factor" with a clear meaning. It is used in papers to compare models and during training
to track progress.

**Worked example: across model generations.**

```text
PPL = exp(average cross-entropy loss)
  GPT-2   (2019): loss ≈ 3.1 → PPL = 22.2   ← ~22 options on average
  GPT-3   (2020): loss ≈ 2.6 → PPL = 13.5
  GPT-4   (2023): loss ≈ 1.8 → PPL =  6.0
  Claude 3 (2024): loss ≈ 1.5 → PPL = 4.5
Human-level on well-formed text: PPL ≈ 2–4.   (Values approximate. Exact figures are proprietary.)
```

**Where it shows up:** evaluating how well a model predicts held-out text (Chapter 17).

### KL divergence (`D_KL(P ‖ Q)`)

KL divergence measures how different two probability distributions are. Given a reference Q and a new
distribution P, `D_KL(P ‖ Q)` is how much extra you'd pay to encode samples from P using a code built
for Q, measured in *bits* if you take log base 2, or *nats* if you use the natural log, which is
what training code does and what the worked example below uses. If P = Q it is 0, and it grows without bound as they diverge. It is *not* symmetric:
`D_KL(P ‖ Q) ≠ D_KL(Q ‖ P)`.

$$D_{\mathrm{KL}}(P\,\Vert\,Q)=\sum_{x} P(x)\,\log\frac{P(x)}{Q(x)}$$

**Why it matters:** It is central to alignment. During RLHF, a KL penalty stops the model drifting
too far from its original behavior while it chases the reward, keeping it from gaming the reward
model at the expense of fluent language. (In the training loss `β·D_KL(π_θ ‖ π_ref)`, π is a model's
output distribution, π_θ the model being aligned and π_ref the frozen original, and β dials how hard
the penalty pushes. Chapters 15 and 16 use exactly this notation.)

**Worked example: two next-token distributions.**

```text
Original model Q:           Aligned model P:
  P("helpful")  = 0.7         P("helpful")  = 0.8
  P("harmless") = 0.2         P("harmless") = 0.15
  P("other")    = 0.1         P("other")    = 0.05

Using natural log, so the answer is in nats (training code always does):

D_KL(P ‖ Q) = 0.8·ln(0.8/0.7) + 0.15·ln(0.15/0.2) + 0.05·ln(0.05/0.1)
            = 0.107 − 0.043 − 0.035
            = 0.029     ← small drift, acceptable
```

**Where it shows up:** the KL penalty in RLHF and related alignment objectives (Chapter 15).

## Summary

- **Linear algebra** is the substrate: vectors and matrices, connected by matrix multiplication,
  itself just a grid of dot products. The dot product measures geometric alignment. Cosine similarity is the
  same idea with length divided out. Transpose and rotation reshape vectors so attention and position
  work.
- **Probability** turns the model's raw scores into choices: softmax makes logits into a distribution,
  and temperature and top-p shape how that distribution is sampled.
- **Optimization** is how the model learns: gradients say which way to nudge each weight,
  backpropagation computes them all at once, and Adam turns them into stable steps sized by a learning
  rate.
- **Functions and numerical tricks** keep it working: the √d_k scaling keeps attention trainable,
  activation functions add the non-linearity that makes depth matter, and dropout guards against
  overfitting.
- **Statistics** (mean, variance) power normalization, and **floating point / quantization** decide
  what fits in memory.
- **Information** measures success: cross-entropy is the training loss, perplexity makes it
  interpretable, and KL divergence keeps an aligned model close to its origin.

## Where the rest of the math lives

A few concepts sit on the border between "mathematical building block" and "architecture component."
Rather than define them twice, this book teaches each one where it is used, and introduces its
mathematics there. You'll find:

- **Tokenization / Byte-Pair Encoding** → Chapter 3 (turning text into token IDs).
- **Positional encoding** → Chapter 4 (built on the *Rotation (RoPE)* primitive above).
- **Multi-head attention and causal masking** → Chapter 5.
- **Residual connections, layer normalization, and RMSNorm** → Chapter 6 (built on *Mean* and
  *Variance* above).
- **Weight tying** → Chapter 8.
- **Mixture of experts** → Chapter 9.

> **Coming up:** These operations describe how numbers move, but not yet where the numbers come
> from. A model's first job is to turn words into the vectors it does all this arithmetic on.
> Chapter 3 begins there: tokenization and embeddings.

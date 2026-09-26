::: {.frontmatter}

# About the Cover

The cover line reads: *Machines learned to read us. Now they answer.* The last word, *answer*, is set in red, and a fan of red arches rises over the line, each one linking a word of the line back to it. The small caps beneath, *what the answer looks back at*, describe those arches exactly. They are the subject of this book, drawn as one picture: what the machine weighs when it arrives at that final word.

## What the arches show

When a language model writes a word, it does not look only at the word just before it. It looks back over everything it has read so far and decides how much each earlier word should count. That looking-back is called *attention*, and it is the idea the whole book is built around.

Each arch connects *answer* to a word in the line, and the thicker the arch, the more that word mattered when the machine reached the end of the sentence. Think of answering a question out loud. Before you speak, your mind leans on the parts of the question that carry the most weight and skims past the rest. The arches draw that leaning.

The words in the line are shaded to match. The darker a word, the more the final word depends on it, and the palest words barely register. You can read the sentence twice: once for its meaning, and once for its color, which shows you where the machine was looking.

## Where the machine looked

The thickest arch is worth a second look. It does not run back to *Machines*, the word you might expect to matter most. It runs to *read*. The sentence is really about the act of reading, not only about who is doing it, so *read* is the word the ending leans on hardest. This is normal for language. We build meaning out of many earlier words in uneven amounts, and the important ones are not always the obvious ones.

There is a fixed budget of attention. If you add up how much *answer* draws on every word in the line, the total comes to exactly one, like slices of a single pie. Every word competes for a slice, so a larger share for one means a smaller share for the rest. Even the tiny arch curling over *answer* itself is part of that budget. A word always keeps a little attention on its own place in the line, however much it borrows from elsewhere.

Notice, too, that the arches sail straight over the period in the middle. Attention does not stop at the end of a sentence. It reaches back as far as it needs to, and here it reaches all the way to the first word.

## Why this is a picture of an answer

No arch reaches forward, and that is not just how the picture was drawn. A model like this works strictly left to right, one word at a time. When it takes in a word, it can weigh everything it has seen so far, that word included, but the words that will follow have not been written yet, so there is nothing ahead to look at.

This is the part worth pausing on. Reading and answering are not two separate skills. The machine produces a reply one word at a time, and it builds each new word out of everything before it, using the exact operation drawn on the cover. The fan is what the model does to understand a sentence, and it is also what the model does to write one. The first half of the cover line, *Machines learned to read us*, describes the reading. The picture explains the second half, *Now they answer*.

## A note on the numbers

The weights on the cover are a careful reading of this one sentence, not a measurement taken from a running model. Everything else in the picture follows from them: the thickness of each arch, the shade of each word, and the fact that they add up to one. The cover shows the shape of attention, not one model's exact opinion of one line.

:::

---
layout: post
title: Markdown语法笔记
published: true
date: 2016-08-05
---

> 博客使用Markdown，记录一下基本语法的使用。


# APPEARANCE

```
# CODE
```

# HEADING

```
# HEADING
```

###### HEADING

```
###### HEADING
```

HEADING
=======

```
HEADING
=======
```

HEADING
-------

```
HEADING
-------
```

<s>DELETED</s>

~~DELETED~~

```
<s>DELETED</s>

~~DELETED~~
```

*ITALIC*

_ITALIC_

```
*ITALIC*

_ITALIC_
```

**BOLD**

__BOLD__

```
**BOLD**

__BOLD__
```

***BOLD ITALIC***

___BOLD ITALIC___

```
***BOLD ITALIC***

___BOLD ITALIC___
```

2H<sub>2</sub> + 0<sub>2</sub> -> 2H<sub>2</sub>0

```
2H<sub>2</sub> + 0<sub>2</sub> -> 2H<sub>2</sub>0
```

A<sup>2</sup> + B<sup>2</sup> = C<sup>2</sup>

```
A<sup>2</sup> + B<sup>2</sup> = C<sup>2</sup>
```

<abbr title="People's Republic of China">PRC</abbr>

```
<abbr title="People's Republic of China">PRC</abbr>
```

> 早
>
> -- <cite>鲁迅</cite>

```
> 早
>
> -- <cite>鲁迅</cite>
```

- POINT ONE

- POINT TWO

- POINT ...

```
- POINT ONE

- POINT TWO

- POINT ...
```

* POINT ONE

* POINT TWO

* POINT ...

```
* POINT ONE

* POINT TWO

* POINT ...
```

+ POINT
    + POINT
    + POINT
+ POINT
    + POINT

```
+ POINT
    + POINT
    + POINT
+ POINT
    + POINT
```

1. POINT ONE
2. POINT TWO
3. POINT ...

```
1. POINT ONE
2. POINT TWO
3. POINT ...
```

- [x] TASK 1
- [ ] TASK 2
    - [x] TASK 2.1
    - [ ] TASK 2.2

```
- [x] TASK 1
- [ ] TASK 2
    - [x] TASK 2.1
    - [ ] TASK 2.2
```
    
`git status`

```
`git status`
```

<https://github.com/wadeee/>

```
<https://github.com/wadeee/>
```


[GITHUB](https://github.com/wadeee/)

```
[GITHUB](https://github.com/wadeee/)
```

[GITHUB][TARGET]

[TARGET]: https://github.com/wadeee/ "wade's github"

```
[GITHUB][TARGET]

[TARGET]: https://github.com/wadeee/ "wade's github"
```


![](/images/posts/learn-markdown/pic.jpg "pic")

```
![](/images/posts/learn-markdown/pic.jpg "pic")
```

[![pic]][pic link]

[pic]: /images/posts/learn-markdown/pic.jpg "pic"

[pic link]: https://github.com/wadeee/

```
[![pic]][pic link]

[pic]: /images/posts/learn-markdown/pic.jpg "pic"

[pic link]: https://github.com/wadeee/
```

    TEXT LIKE `<pre>`

```
    TEXT LIKE `<pre>`
```

```html
<!DOCTYPE html>
<html>
    <head>
        <mate charest="utf-8" />
        <title>Hello world!</title>
    </head>
    <body>
        <h1>Hello world!</h1>
    </body>
</html>
```


```html
    ```html 
    <!DOCTYPE html>
    <html>
        <head>
            <mate charest="utf-8" />
            <title>Hello world!</title>
        </head>
        <body>
            <h1>Hello world!</h1>
        </body>
    </html>
    ``` 
```

| Function name | Description                    |
| ------------- | ------------------------------ |
| `help()`      | Display the help window.       |
| `destroy()`   | **Destroy your computer!**     |

```
| Function name | Description                    |
| ------------- | ------------------------------ |
| `help()`      | Display the help window.       |
| `destroy()`   | **Destroy your computer!**     |
```

---
----
***
*****

```
    ---
    ----
    ***
    *****
```
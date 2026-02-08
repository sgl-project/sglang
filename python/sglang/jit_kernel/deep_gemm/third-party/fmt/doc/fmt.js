document$.subscribe(() => {
  hljs.highlightAll(),
  { language: 'c++' }
})

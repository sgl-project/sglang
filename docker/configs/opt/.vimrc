function! Yank(text) abort
  let escape = system('yank', a:text)
  if v:shell_error
    echoerr escape
  else
    call writefile([escape], '/dev/tty', 'b')
  endif
endfunction

noremap <silent> <Leader>y y:<C-U>call Yank(@0)<CR>

" automatically run yank(1) whenever yanking in Vim
function! CopyYank() abort
  call Yank(join(v:event.regcontents, "\n"))
endfunction

autocmd TextYankPost * call CopyYank()

" Basic settings
set number
syntax on
set mouse=a
filetype indent on

" Indentation
set autoindent nosmartindent
set smarttab
set expandtab
set shiftwidth=4
set softtabstop=4

" Visual guides
set colorcolumn=120
highlight ColorColumn ctermbg=5

" Status line
set laststatus=2
set statusline=%<%f\ %h%m%r%=%{\"[\".(&fenc==\"\"?&enc:&fenc).((exists(\"+bomb\")\ &&\ &bomb)?\",B\":\"\").\"]\ \"}%k\ %-14.(%l,%c%V%)\ %P

" Backspace behavior
set backspace=2

" Encoding
set encoding=utf-8
set fileencoding=utf-8

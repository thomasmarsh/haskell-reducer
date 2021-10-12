all: README.html README.md

clean:
	$(RM) README.html README.md tmp.md

README.html: app/Main.lhs
	pandoc -s app/Main.lhs -o README.html --metadata title="Reducer Architecture in Haskell"

README.md: app/Main.lhs
	pandoc -s app/Main.lhs -o tmp.md --metadata title="Reducer Architecture in Haskell"
	grep -v '^---$$' tmp.md | sed -e 's/``` {.*/```haskell/; s/^title: /# /;' > README.md
	rm tmp.md

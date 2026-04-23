git checkout -b add-<model-slug>
git add docs_new/cookbook/<category>/<Vendor>/<Model>.mdx
git add docs_new/src/snippets/autoregressive/<name>-deployment.jsx
git add docs_new/cookbook/<category>/intro.mdx
git add docs_new/cookbook/<category>/<Vendor>/<PreviousNew>.mdx
git add docs_new/docs.json
git commit -m "Add <Model Name> cookbook"
git push -u origin add-<model-slug>
gh pr create --title "Add <Model Name> cookbook" --body "..."

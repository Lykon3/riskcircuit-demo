# Git Commands and Workflows for Beginners

## 📚 Table of Contents
- [Getting Started](#-getting-started)
- [Basic Git Commands](#-basic-git-commands)
- [Common Workflows](#-common-git-workflows)
- [GitHub Actions](#-automated-workflows-with-github-actions)
- [Useful Commands](#-useful-git-commands)
- [Best Practices](#-best-practices)
- [Common Issues](#-common-issues-and-solutions)
- [Aliases](#-helpful-aliases)
- [Next Steps](#-next-steps)
- [Resources](#-resources)

## 🚀 Getting Started
```bash
# Configure your identity (one-time setup)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## 📁 Basic Git Commands
```bash
# Create a new repository
git init

# Clone an existing repository
git clone https://github.com/username/repository.git

# Check status of your files
git status

# Add files to staging area
git add filename.txt
git add .

# Commit changes
git commit -m "Describe what you changed"

# Push changes to GitHub
git push origin main

# Pull latest changes from GitHub
git pull origin main
```

## 🔄 Common Git Workflows

### 1. Basic Daily Workflow
```bash
git pull
# Make changes
git status
git add .
git commit -m "Changes made"
git push
```

### 2. Working with Branches
```bash
git checkout -b feature-name
git branch
git checkout main
git merge feature-name
git branch -d feature-name
```

## 🤖 Automated Workflows with GitHub Actions

### CI/CD
Create `.github/workflows/main.yml`:
```yaml
name: CI/CD Pipeline
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
    - name: Install dependencies
      run: npm install
    - name: Run tests
      run: npm test
```

### Auto-format Code
```yaml
name: Auto Format
on:
  push:
    branches: [ main ]
jobs:
  format:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Format code
      run: |
        pip install black
        black .
    - name: Commit changes
      run: |
        git config --local user.email "action@github.com"
        git config --local user.name "GitHub Action"
        git add -A
        git diff --staged --quiet || git commit -m "Auto-format code"
        git push
```

## 🛠️ Useful Git Commands
```bash
git log --oneline
git diff filename.txt
git blame filename.txt
git checkout -- filename.txt
git reset HEAD filename.txt
git reset --soft HEAD~1
git reset --hard HEAD~1
git stash
git stash list
git stash pop
```

## 💡 Best Practices
1. Commit often
2. Use clear messages
3. Pull before push
4. Use branches
5. Review changes with `git diff`

## 🚨 Common Issues and Solutions
### Merge Conflicts
```bash
# Open conflicted files
# Fix the code
# Remove conflict markers
git add .
git commit -m "Resolved conflicts"
```

### Push Rejected
```bash
git pull origin main
# Fix conflicts
git push origin main
```

## 📚 Helpful Aliases
```ini
[alias]
    st = status
    co = checkout
    br = branch
    cm = commit -m
    ps = push
    pl = pull
    lg = log --oneline --graph --all
```

## 🎯 Next Steps
1. Practice: create and push a repo
2. Use GitHub Desktop
3. Learn rebasing and cherry-picking
4. Contribute to open source

## 📖 Resources
- https://guides.github.com/introduction/git-handbook/
- https://learngitbranching.js.org/
- https://docs.github.com/en/actions
- https://www.atlassian.com/git/tutorials

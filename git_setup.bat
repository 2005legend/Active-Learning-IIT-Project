@echo off
echo Setting up Git repository for Policy Gradient Active Learning...
echo.

REM Initialize git repository
git init

REM Add all files
echo Adding files to git...
git add .

REM Create initial commit
echo Creating initial commit...
git commit -m "Initial commit: Policy Gradient Active Learning project

- Baseline CNN training (95.22% accuracy)
- Active Learning comparison (Random vs Uncertainty)
- REINFORCE policy gradient implementation (93.78% accuracy)
- Interactive Streamlit UI with AI explanations
- Comprehensive documentation and analysis
- 75% data reduction with minimal performance loss"

echo.
echo Git repository initialized successfully!
echo.
echo Next steps:
echo 1. Create a new repository on GitHub
echo 2. Copy the repository URL
echo 3. Run: git remote add origin [your-repo-url]
echo 4. Run: git branch -M main
echo 5. Run: git push -u origin main
echo.
echo Your project is ready for GitHub!
pause
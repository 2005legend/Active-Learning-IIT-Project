#!/usr/bin/env python3
"""
Validation script to ensure project is ready for GitHub upload
"""

import os
import subprocess
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and report status"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - MISSING")
        return False

def check_git_status():
    """Check git repository status"""
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            if result.stdout.strip() == "":
                print("✅ Git: All files committed")
                return True
            else:
                print("⚠️  Git: Uncommitted changes found")
                print(result.stdout)
                return False
        else:
            print("❌ Git: Repository not initialized")
            return False
    except FileNotFoundError:
        print("❌ Git: Git not installed or not in PATH")
        return False

def check_file_sizes():
    """Check for large files that shouldn't be in git"""
    large_files = []
    for root, dirs, files in os.walk('.'):
        # Skip .git directory
        if '.git' in dirs:
            dirs.remove('.git')
        
        for file in files:
            filepath = os.path.join(root, file)
            try:
                size = os.path.getsize(filepath)
                if size > 50 * 1024 * 1024:  # 50MB
                    large_files.append((filepath, size))
            except OSError:
                continue
    
    if large_files:
        print("⚠️  Large files found (>50MB):")
        for filepath, size in large_files:
            print(f"   {filepath}: {size / (1024*1024):.1f}MB")
        return False
    else:
        print("✅ File sizes: No large files found")
        return True

def main():
    """Main validation function"""
    print("🔍 Validating GitHub readiness...")
    print("=" * 50)
    
    checks = []
    
    # Essential files
    checks.append(check_file_exists("README_GITHUB.md", "Main README"))
    checks.append(check_file_exists("LICENSE", "License file"))
    checks.append(check_file_exists("CONTRIBUTING.md", "Contributing guidelines"))
    checks.append(check_file_exists(".gitignore", "Gitignore file"))
    checks.append(check_file_exists("requirements.txt", "Requirements file"))
    
    # Core source files
    checks.append(check_file_exists("app.py", "Streamlit app"))
    checks.append(check_file_exists("src/", "Source code directory"))
    checks.append(check_file_exists("src/models/cnn.py", "CNN models"))
    checks.append(check_file_exists("src/al/strategies.py", "Active learning strategies"))
    checks.append(check_file_exists("src/rl/policy.py", "REINFORCE policy"))
    
    # Documentation
    checks.append(check_file_exists("GITHUB_SETUP_CHECKLIST.md", "Setup checklist"))
    checks.append(check_file_exists("FINAL_GITHUB_STEPS.md", "Final steps guide"))
    
    # Git status
    checks.append(check_git_status())
    
    # File sizes
    checks.append(check_file_sizes())
    
    print("\n" + "=" * 50)
    
    if all(checks):
        print("🎉 SUCCESS: Project is ready for GitHub upload!")
        print("\n📋 Next steps:")
        print("1. Create repository on GitHub")
        print("2. git remote add origin <your-repo-url>")
        print("3. git branch -M main")
        print("4. git push -u origin main")
        print("\n🚀 Your research project is ready to showcase!")
    else:
        print("⚠️  ISSUES FOUND: Please resolve the above issues before uploading")
        failed_checks = sum(1 for check in checks if not check)
        print(f"   {failed_checks} checks failed out of {len(checks)}")

if __name__ == "__main__":
    main()
# CSE Minor project files

Workflow:

1. **Get started.** Copy the repository files on your local machine by using: ```git clone https://github.com/gcalin/cseminor.git```.
   - **Authentication**. Git will require you to log in to github.com when doing different operations, so have to make an account first. Additionally, IDEs like Visual Studio Code and all the JetBrains products will have integrated functionality with git, so you may use that if you prefer.

2. **Updated the repository.** In order to see what changes have been made to the repository since the last time you updated, you can use ```git pull```. Make sure to always updated the repository before making any changes, in order to avoid conflicts!
   
3. **Make changes.** Once your repository is up to date, you can start adding, deleting and changing files. Make the changes on your local machine, and once you are happy with them, push them to the repository with the following commands.
    - **See what changed.** To see all the changes you made to the repository, use ```git status```. This will show you all the information regarding the files you added, deleted or changed.
    - **Add the files.** After seeing what you changed, you need to decide which of the changed files you want to push to the repository. The first step to do this is to add the files to a local buffer before commiting them to the repository. You can do this by using ```git add <file or directory name> <file or directory name>  ...```. You can add files one by one, your can add all the changed files by using ```git add -A```. You do not need to add all the files that you changed, and creating separate commits for unrelated files is encouraged since it is easier to track.
    - **Commit your files.** After adding your files, you need to "package" those files in a container, called a commit. To "label" this "package", you need to provide a message that very briefly explains what changed. You can do this with ```git commit -m "This is what I changed and why I did it..."```. You can see a history of commits with ```git log```.
    - **Push your commit(s).** At this point, you may notice that despite ```git commit```-ing all your files, your changes are still not visible on github, or for the other contributors. This is because a commiting only packages changes, and does not add them to the remot repository, but rather to your local copy. In order to add them to the remore repository, you can use the command ```git push origin main```. Side note: Pushing directly to the main branch is generally a very bad idea, but for small scale projects like this one, it is more convenient.
    
4. **Solve conflicts.** It may sometime happen that ```git push```-ing your commit(s) requires you to ```pull``` (update) your repository first or results in conflicts, if you and some other person changed the sime line in a file, and the other person ```push```ed first. In this case, it is best to ```git pull``` first and fix the conflicts locally, and then ```commit``` and ```push``` these changes as you normally would any other change. Do not be afraid to fix conflicts, git tracks all commits and reverting can be done easily, if need be.



Video tutorials: [this video by one of the professors from EWI](https://www.youtube.com/watch?v=Hm9wTx6p90k) does a good job of explaining exactly what we need for this particular project. For a more popular tutorial that also discusses branches, you can check out [this](https://www.youtube.com/watch?v=SWYqp7iY_Tc) video instead.

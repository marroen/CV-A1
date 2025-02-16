import project

def main():
    #run = 25
    #run = 10
    run = 5

    project.get_all_points(run=run)
    project.project_cube(run=run, webcam=False)
  
if __name__ == "__main__":
    main()
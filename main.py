import turtle
import time

def main():
    with open('./something.logo') as file:
        for line in file.readlines():
            print(line)

if __name__ == '__main__':
    main()
import re

def main():
    s = 'abcaaaa'
    
    for mo in re.finditer('abcaaaa|abc', s):
        print(mo)

if __name__ == '__main__':
    main()
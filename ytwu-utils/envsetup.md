# 换机器流程

1. gemini上手动执行
```bash
apt update
apt install -y openssh-server
echo -e "PermitRootLogin yes\nPasswordAuthentication yes\nPort 2222" | tee -a /etc/ssh/sshd_config > /dev/null
service ssh restart
echo "root:123456wyt" | chpasswd

ssh-keygen
ssh-copy-id -p 2222 127.0.0.1
```



2. 配置ssh config：在跳板机上执行（记得修改IP）
```bash
MASTER=29.225.114.43
SLAVE=29.225.99.178

rm -rf ~/.ssh/config

echo -e "
Host ytn1
  Hostname $MASTER
  User root
  Port 2222

  Host ytn2
  Hostname $SLAVE
  User root
  Port 2222
" > ~/.ssh/config
ssh-copy-id ytn1
ssh-copy-id ytn2
scp ~/.ssh/id_rsa ytn1:/root/.ssh/id_rsa

# map ytn1 to MASTER to /etc/hosts
echo "$MASTER ytn1" >> /etc/hosts
echo "$SLAVE ytn2" >> /etc/hosts
```

3. 在ytn1, ytn2机器上配置ift
待自动化

ift sync使用方法，将开发机的~/ytwu/save/sglang/与部署机的/sgl-workspace/sglang/同步

```bash
# 在开发机：
cd ~/ytwu/
ft agent -k

# 在不方便vscode上去的部署机
ift_i="$HOME/ift_install.sh"; export PATH="$PATH:$HOME/.ft"; for i in 9.139.66.141 9.139.66.133 9.139.66.142 10.28.37.11 10.28.37.12 10.99.245.206 10.99.245.207 10.123.119.236 10.123.119.237 9.135.114.22 9.135.114.23 9.218.224.68 9.218.224.80; do curl -v -fksSL --connect-timeout 1 --noproxy '*' -o "$ift_i" http://$i/install && break; done ; test -s "$ift_i" && bash "$ift_i" || echo failed to install ft
cd /
ft syncd -u yongtongwu -d VM-162-200-tencentos /sgl-workspace

rm -f ~/.ft/rsync/ft.rsync
ln -s $(which rsync) ~/.ft/rsync/ft.rsync

while true; do
    echo "Starting ft syncd sgl-workspace"
    ft syncd -u yongtongwu -d VM-162-200-tencentos /sgl-workspace
done
```

4. 执行setenv.sh
待自动化


5. 待sglang pull完毕
```bash
proxy_off
pip3 install --trusted-host mirrors.cloud.tencent.com -e "python[all]"
```


# Git 指定密钥
```bash
GIT_SSH_COMMAND="ssh -i ~/ytwu/.ssh/id_ed25519"  git push --set-upstream  my-gh 250311/mtp-profile
```

# benchmark tool打包

```bash
python setup.py bdist_wheel
pip install twine
twine upload dist/* --repository-url https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple --username yongtongwu --password d2ce379cff1711efa4ce5254009bad14

pypi-token获取：
https://mirrors.tencent.com/#/private/pypi
->获取访问token
```
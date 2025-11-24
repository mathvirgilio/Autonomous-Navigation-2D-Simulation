def sphere(Chrom):
    acc = 0
    dim = Chrom.size
    for i in range(dim):
        acc = acc + Chrom[i]**2
    return acc

def quadric(Chrom):
    acc_2 = 0
    dim = Chrom.size
    for i in range(dim):
        acc_1 = 0
        for j in range(i):
            acc_1 = acc_1 + Chrom[i]
        acc_2 = acc_2 + acc_1**2
    return acc_2

def rosenbrock(Chrom):
    acc = 0
    dim = Chrom.size
    for i in range(dim//2):
        acc = acc + 100*(Chrom[2*i]-Chrom[2*i-1]**2)**2+(1-Chrom[2*i-1])**2;
    return acc

'''
function ObjVal = rastrigin(Chrom)

% Dimension of objective function
    Dim=size(Chrom,2);
   
% Compute population parameters
      A = 10;
      Omega = 2 * pi;
      ObjVal = Dim * A + sum(((Chrom .* Chrom) - A * cos(Omega * Chrom))')';


function ObjVal = schwefel(Chrom,switch1);

% Dimension of objective function
  
    Dim=size(Chrom,2);
   
% Compute population parameters
   [Nind,Nvar] = size(Chrom);


      % function 7, sum of -xi*sin(sqrt(abs(xi))) for i = 1:Dim (Dim=10)
      % n = Dim, -500 <= xi <= 500
      % global minimum at (xi)=(420.9687) ; fmin=?
      ObjVal = 418.9829*Dim - sum((Chrom .* sin(sqrt(abs(Chrom))))')';
   % otherwise error, wrong format of Chrom

function ObjVal = ackley(Chrom,switch1);

Dim=size(Chrom,2);
   
% Compute population parameters
   [Nind,Nvar] = size(Chrom);


      A = 1/Dim;
      Omega = 2 * pi;
      sum1=A.*sum((Chrom .* Chrom)')';
      %sum1=A.*sum(Chrom .* Chrom);
      sum2=A.*sum((cos(Omega * Chrom))')';
      ObjVal = -20*exp(-0.2*sqrt(sum1))-exp(sum2)+20+exp(1);
'''
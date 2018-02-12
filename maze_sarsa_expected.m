
%probabilities of actions
a=0.9;
b=0.05

%state
S=1:16;
goal=16;
bad_state=10;
initial_state=5;
s_init=initial_state;

%Reward matrix
R=ones(16,16);
R=R*nan;

R(1,2)=-1, R(1,5)=-1, R(2,1)=-1, R(2,3)=-1, R(2,6)=-1, R(3,2)=-1, R(3,4)=-1, R(3,7)=-1, R(4,3)=-1, R(4,8)=-1,
R(5,1)=-1,R(5,6)=-1,R(5,9)=-1,R(6,5)=-1,R(6,2)=-1,R(6,7)=-1,R(6,10)=-1,R(7,6)=-1,R(7,3)=-1,R(7,11)=-1,
R(7,8)=-1,R(8,7)=-1,R(8,4)=-1,R(8,12)=-1,R(9,5)=-1,R(9,10)=-1,R(9,13)=-1,R(10,9)=-1,R(10,6)=-1,
R(10,11)=-1,R(10,14)=-1,R(11,10)=-1,R(11,7)=-1,R(11,15)=-1,R(11,12)=-1,R(12,11)=-1,R(12,8)=-1,R(12,16)=-1,
R(13,9)=-1,R(13,14)=-1,R(14,13)=-1,R(14,10)=-1,R(14,15)=-1,R(15,11)=-1,R(15,11)=-1,R(16,15)=-1,
R(16,12)=-1,R(1,5)=0,R(9,5)=0,R(6,5)=0,R(6,10)=-70,R(9,10)=-70,R(14,10)=-70,R(11,10)=-70,R(12,16)=100,
R(15,16)=100;

%action-value matrix
Q=zeros(16,16);

%count of each action from all states N(s,a)
N=zeros(16,16);

%hyperparameter
gamma=0.99;
epsilon1=0;
epsilon2=0.2;
bad_count=0;

episodes=6000;
avg_reward_expected=zeros(1,episodes);
iter=50
states_visited_eps=zeros(1,episodes);

for runs=1:iter
     episode_count=0;
for episode=(1:episodes)
    episode_reward=0;
    states_visited=0;
    s_init=5;
    
    while(s_init~=goal)
        s_init
        states_visited=states_visited+1
        if(s_init==10)
            break;
        end
    
        possible_actions=[];
        possible_actions_i=[];
        for i=1:16
            if (~isnan(R(s_init,i)))
                possible_actions=[possible_actions,i];
            end
        end
    
        l=length(possible_actions);
        q_val=[];
        rnd=0.01*(randi(101)-1);
        if(rnd>epsilon1)
            for i=1:l
                q_val=[q_val,Q(s_init,possible_actions(i))];
            end
            [m,index]=max(q_val);
            action=possible_actions(index);
        else
            index=randi(l);
            action=possible_actions(index);
        end
    
        i_state=action;
        
        
        for i=1:16
            if (~isnan(R(i_state,i)))
                possible_actions_i=[possible_actions_i,i];
            end
        end
       
        action_i=possible_actions_i;
        q_next=[];
       l1=length(action_i);
       for j=(1:l1)
           q_next=[q_next,Q(i_state,action_i(j))];
       end
        v_s=0
        l_action=length(action_i)
        for i=1:l_action
            [m,index]=max(q_next)
            act=action_i(index);
            if(action_i(i)==act)
                pi=(epsilon1/l1)+1-epsilon1
            else
                pi=(epsilon1/l1);
            end
            v_s=v_s+Q(i_state,action_i(i))*pi
            
            
        end
        N(s_init,i_state)= N(s_init,i_state)+1;
        alpha=1/N(s_init,i_state);
        Q(s_init,i_state)=Q(s_init,i_state)+alpha*(R(s_init,i_state)+gamma*v_s-Q(s_init,i_state))
        episode_reward=episode_reward+R(s_init,i_state);
        s_init=i_state;
    end
    
    states_visited_eps(1,episode_count)=states_visited_eps(1,episode_count)+states_visited;
    avg_reward_expected(1,episode_count)=avg_reward_expected(1,episode_count)+episode_reward;
    
end
end

avg_reward_expected=avg_reward_expected/iter
states_visited_eps=states_visited_eps/iter

plot(avg_reward_expected)
figure;
plot(states_visited_eps)


avg_reward_eps10=[]
states_visited_eps10=[]
s=[],s_count=0;
s_states=[]
 for i=1:episodes
     i
     s_count=s_count+1;
     
     s=[s,avg_reward_expected(1,i)];
     s_states=[s_states,states_visited_eps(1,i)]
     if(s_count==20)
         s_count=0;
         f1=mean(s)
         avg_reward_eps10=[avg_reward_eps10,f1];
         
         f2=mean(s_states)
         states_visited_eps10=[states_visited_eps10,f2];
         s=[];
         s_states=[];
     end
 end
figure;
plot(avg_reward_eps10)
ylim=([1 100])
figure;
plot(states_visited_eps10)
ylim=([1 100])

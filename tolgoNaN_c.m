function [tracks_nonan]=tolgoNaN_c(tracks)

tracks_nonan=tracks;
Indici=zeros(size(tracks));
Indici=isnan(tracks);
Diffe=diff(Indici);
[frames_kin,variabili]=size(tracks);
ind_in=[]; ind_fin=[];
%interpolo cinematica for missing frames
for k=1:variabili
    if (Indici(1,k)==1)
       ind_in(1,1)=1; 
    end
    ind_in=[ind_in;find(Diffe(:,k)==1)];
    ind_fin=find(Diffe(:,k)==-1);
    if (isempty(ind_in)==0 && ind_in(1)==1)
       tracks_nonan(1:ind_fin(1)+1,k)=tracks(ind_fin(1)+1,k); %assegno a tutti i primi NaN il valore del primo frame che acquisisco
       if length(ind_in)==length(ind_fin)
            for j=2:length(ind_in)
                y=[tracks(ind_in(j),k);tracks(ind_fin(j)+1,k)];
                tracks_nonan(ind_in(j):ind_fin(j)+1,k)=interp1([1,2],y,linspace(1,2,ind_fin(j)-ind_in(j)+2),'spline');
            end
        else
            for j=2:length(ind_fin)
               y=[tracks(ind_in(j),k);tracks(ind_fin(j)+1,k)];
                tracks_nonan(ind_in(j):ind_fin(j)+1,k)=interp1([1,2],y,linspace(1,2,ind_fin(j)-ind_in(j)+2),'spline');
            end
            tracks_nonan(ind_in(end):end,k)=tracks(ind_in(end)-1,k).*ones(size(tracks_nonan(ind_in(end):end,k)));
        end
    elseif (isempty(ind_in)==0 && ind_in(1)>1)%se il primo frame non è NaN
        if length(ind_in)==length(ind_fin)
            for j=1:length(ind_in)
                y=[tracks(ind_in(j),k);tracks(ind_fin(j)+1,k)];
                tracks_nonan(ind_in(j):ind_fin(j)+1,k)=interp1([1,2],y,linspace(1,2,ind_fin(j)-ind_in(j)+2),'spline');
            end
            if(length(ind_fin)>0)
                for j=1:length(ind_fin)
                    y=[tracks(ind_in(j),k);tracks(ind_fin(j)+1,k)];
                    tracks_nonan(ind_in(j):ind_fin(j)+1,k)=interp1([1,2],y,linspace(1,2,ind_fin(j)-ind_in(j)+2),'spline');
                end
            end
            tracks_nonan(ind_in(end):end,k)=tracks(ind_in(end)-1).*ones(1,size(tracks_nonan(ind_in(end):end,k),1));
        elseif length(ind_in>length(ind_fin))
            for j=1:length(ind_in)-1
                y=[tracks(ind_in(j),k);tracks(ind_fin(j)+1,k)];
                 tracks_nonan(ind_in(j):ind_fin(j)+1,k)=interp1([1,2],y,linspace(1,2,ind_fin(j)-ind_in(j)+2),'spline');
            end
            if(length(ind_fin)>0)
                for j=1:length(ind_fin)
                    y=[tracks(ind_in(j),k);tracks(ind_fin(j)+1,k)];
                    tracks_nonan(ind_in(j):ind_fin(j)+1,k)=interp1([1,2],y,linspace(1,2,ind_fin(j)-ind_in(j)+2),'spline');
                end
            end
            tracks_nonan(ind_in(end):end,k)=tracks(ind_in(end)-1).*ones(1,size(tracks_nonan(ind_in(end):end,k),1));

            tracks_nonan(ind_in(end):end,k)=tracks(ind_in(end),k)*ones(size(tracks_nonan(ind_in(end):end,k)));
        end
    end
    ind_in=[]; ind_fin=[];
end


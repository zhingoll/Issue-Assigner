(function() {
  console.log('Content script loaded');

  // Check if the current page is an Issue page
  const isIssuePage = () => {
    return /^\/[^\/]+\/[^\/]+\/issues\/\d+$/.test(window.location.pathname);
  };

  console.log('Current pathname:', window.location.pathname);
  console.log('Is issue page:', isIssuePage());

  // Get the user login name
  function getUserLogin() {
    const meta = document.querySelector('meta[name="user-login"]');
    if (meta) {
      return meta.getAttribute('content');
    } else {
      return null;
    }
  }

  function addResolverButton() {
    // First check if the current page is an issue page
    if (!isIssuePage()) {
      console.log('Not an issue page');
      return;
    }

    console.log('Adding resolver button');

    // Avoid adding the button multiple times
    if (document.getElementById('issue-resolver-button')) return;

    const button = document.createElement('button');
    button.id = 'issue-resolver-button';
    button.innerText = 'View possible resolvers for this issue';
    button.style.marginLeft = '10px';
    button.style.position = 'relative';
    button.style.zIndex = '1000'; 
    button.classList.add('btn', 'btn-sm', 'btn-primary'); // Using GitHub's btn-primary style

    const actionsBar = document.querySelector('.gh-header-actions');
    if (actionsBar) {
      console.log('Found actions bar');
      actionsBar.appendChild(button);
    } else {
      console.log('Actions bar not found, trying header');
      const header = document.querySelector('.gh-header-show');
      if (header) {
        header.appendChild(button);
      } else {
        console.log('Header not found');
        return;
      }
    }

    button.addEventListener('click', () => {
      console.log('Button clicked');
      const pathParts = location.pathname.split('/');
      const owner = pathParts[1];
      const repo = pathParts[2];
      const issueNumber = pathParts[4];

      console.log('Owner:', owner, 'Repo:', repo, 'Issue Number:', issueNumber);

      const requestData = {
        owner: owner,
        name: repo,
        number: parseInt(issueNumber)
      };

      fetch('http://localhost:8000/get_issue_resolvers', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
      })
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(data => {
        if (data && data.recommendations && data.recommendations.length > 0) {
          showResolvers(data.recommendations, owner, repo, issueNumber);
        } else {
          alert('No possible resolvers found.');
        }
      })
      .catch(error => {
        console.error('Error fetching issue resolvers:', error);
        alert('Error fetching possible resolvers.');
      });
    });
  }

  function showResolvers(recommendations, owner, repo, issueNumber) {
    console.log('Showing resolvers');
    let container = document.getElementById('issue-resolver-container');
    if (container) {
      container.remove();
    }

    container = document.createElement('div');
    container.id = 'issue-resolver-container';
    container.style.marginBottom = '25px';
    container.style.padding = '15px';
    container.style.border = '1px solid #d1d5da';
    container.style.borderRadius = '6px';
    container.style.backgroundColor = '#f6f8fa';
    container.style.display = 'flex';
    container.style.flexDirection = 'row';
    container.style.alignItems = 'flex-start';
    container.style.gap = '20px'; 

    // Create the left container (model selector and recommendations)
    const leftContainer = document.createElement('div');
    leftContainer.style.flex = '0.8'; 
    leftContainer.style.minWidth = '200px';
    leftContainer.style.display = 'flex';
    leftContainer.style.flexDirection = 'column';
    leftContainer.style.gap = '10px';

    // Model selector label and selector
    const modelSelectorLabel = document.createElement('label');
    modelSelectorLabel.innerText = 'Select Model: ';
    modelSelectorLabel.style.fontWeight = 'bold';
    modelSelectorLabel.htmlFor = 'model-selector';

    const modelSelector = document.createElement('select');
    modelSelector.id = 'model-selector';
    modelSelector.style.padding = '5px';
    modelSelector.classList.add('form-control'); // Using GitHub's form-control style

    recommendations.forEach((rec, index) => {
      const option = document.createElement('option');
      option.value = index;
      option.text = rec.model;
      modelSelector.appendChild(option);
    });

    // Add model selector label and selector to left container
    leftContainer.appendChild(modelSelectorLabel);
    leftContainer.appendChild(modelSelector);

    // Recommendations display area
    const resolverDisplayArea = document.createElement('div');
    resolverDisplayArea.id = 'resolver-display-area';
    resolverDisplayArea.style.flex = '1'; 
    resolverDisplayArea.style.paddingTop = '10px';
    resolverDisplayArea.style.borderTop = '1px solid #d1d5da';

    leftContainer.appendChild(resolverDisplayArea);

    container.appendChild(leftContainer);

    // Create right container (stacked bar chart)
    const chartDisplayArea = document.createElement('div');
    chartDisplayArea.id = 'chart-display-area';
    chartDisplayArea.style.flex = '1';
    chartDisplayArea.style.minWidth = '300px';
    chartDisplayArea.style.boxSizing = 'border-box';
    // chartDisplayArea.style.paddingTop= '0px';

    container.appendChild(chartDisplayArea);

    // Add to page
    const discussionTimeline = document.querySelector('.js-discussion');
    if (discussionTimeline) {
      discussionTimeline.parentNode.insertBefore(container, discussionTimeline);
    } else {
      const header = document.querySelector('.gh-header');
      if (header) {
        header.parentNode.insertBefore(container, header.nextSibling);
      }
    }

    // Event listeners
    modelSelector.addEventListener('change', () => {
      const selectedIndex = modelSelector.value;
      displayRecommendation(recommendations[selectedIndex], owner, repo, issueNumber);
    });

    // Default display first model's result
    displayRecommendation(recommendations[0], owner, repo, issueNumber);
  }

  function displayRecommendation(recommendation, owner, repo, issueNumber) {
    const resolverDisplayArea = document.getElementById('resolver-display-area');
    const chartDisplayArea = document.getElementById('chart-display-area');
    if (!resolverDisplayArea || !chartDisplayArea) return;

    // Clear previous content
    resolverDisplayArea.innerHTML = '';
    chartDisplayArea.innerHTML = '';

    // Create recommendation result list
    const resolverContainer = document.createElement('div');
    // resolverContainer.style.marginBottom = '20px';

    const title = document.createElement('h3');
    title.innerText = `Model: ${recommendation.model} recommends to you:`;
    title.style.marginBottom = '10px';
    resolverContainer.appendChild(title);

    const list = document.createElement('ul');
    list.style.listStyleType = 'none';
    list.style.paddingLeft = '20px';
    resolverContainer.appendChild(list);

    // recommendation.assignee.forEach((assignee, index) => {
    //   const item = document.createElement('li');
    //   item.style.marginBottom = '5px';
    //   item.style.display = 'flex';
    //   item.style.alignItems = 'center';

    //   const link = document.createElement('a');
    //   link.href = `https://github.com/${assignee}`;
    //   link.target = '_blank';
    //   link.innerText = assignee;
    //   link.style.fontWeight = '600';
    //   link.style.textDecoration = 'none';
    //   link.style.color = '#0366d6';
    //   link.style.marginRight = '10px';

    //   const probability = recommendation.probability[index];

    //   const probSpan = document.createElement('span');
    //   probSpan.innerText = `(Probability: ${(probability * 100).toFixed(2)}%)`;
    //   probSpan.style.color = '#586069';
    //   probSpan.style.fontSize = '14px';

    //   item.appendChild(link);
    //   item.appendChild(probSpan);
    //   list.appendChild(item);
    // });
    recommendation.assignee.forEach((assignee, index) => {
      const item = document.createElement('li');
      item.style.marginBottom = '10px';
      // Apply the common styles for the list items
      item.style.display = 'block';
      item.style.lineHeight = '26px';
      item.style.position = 'relative';
      // item.style.margin = '3px 0 0 14px';
      item.style.marginLeft = '14px';
      item.style.paddingLeft = '38px';
      item.style.overflow = 'hidden';
      item.style.wordWrap = 'normal';
      item.style.whiteSpace = 'nowrap';
      item.style.textOverflow = 'ellipsis';
      
      // Apply the numbering before pseudo-element equivalent
      const numCircle = document.createElement('span');
      numCircle.style.color = '#000';
      numCircle.style.width = '21px';
      numCircle.style.height = '21px';
      numCircle.style.lineHeight = '20px';
      numCircle.style.textAlign = 'center';
      numCircle.style.position = 'absolute';
      numCircle.style.left = '0';
      numCircle.style.top = '5px';
      numCircle.style.borderRadius = '100%';
      numCircle.style.fontSize = '13px';
      numCircle.style.backgroundColor = '#ececec';
      numCircle.style.textShadow = '0 1px 0 rgba(255,255,255,.5)';
      numCircle.style.fontFamily = 'SourceCodeProRegular, Menlo, Monaco, Consolas, "Courier New", monospace, Helvetica Neue, Arial, sans-serif';
      numCircle.textContent = index + 1;  // Set the number
      item.appendChild(numCircle);
      
      // Apply the special color styles for the top 3 items
      if (index === 0) {
          numCircle.style.backgroundColor = '#e24d46';  // First item
          numCircle.style.color = '#fff';
      } else if (index === 1) {
          numCircle.style.backgroundColor = '#2ea7e0';  // Second item
          numCircle.style.color = '#fff';
      } else if (index === 2) {
          numCircle.style.backgroundColor = '#6bc30d';  // Third item
          numCircle.style.color = '#fff';
      }

      const link = document.createElement('a');
      link.href = `https://github.com/${assignee}`;
      link.target = '_blank';
      link.innerText = assignee;
      // 设置默认字体颜色为黑色
      link.style.color = 'black';
      link.style.fontSize = '16px';

      const probability = recommendation.probability[index];

      const probSpan = document.createElement('span');
      probSpan.innerText = `(Probability: ${(probability * 100).toFixed(2)}%)`;
      probSpan.style.marginLeft = '10px';
      probSpan.style.color = '#888';

      // Hover effect: change text color and underline
      link.addEventListener('mouseenter', () => {
        link.style.textDecoration = 'underline';
        link.style.color = '#1E90FF';
      });
      link.addEventListener('mouseleave', () => {
          link.style.textDecoration = 'none';
          link.style.color = 'black';
      });

      item.appendChild(link);
      item.appendChild(probSpan);

      list.appendChild(item);

      

    
   });
    resolverDisplayArea.appendChild(resolverContainer);

    // Fetch developer statistics and display as a stacked bar chart (only global_openrank, community_openrank, avg_activity)
    fetch('http://localhost:8000/get_developer_stats', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        owner: owner,
        name: repo,
        developers: recommendation.assignee
      })
    })
    .then(res => {
      if (!res.ok) throw new Error('Failed to fetch developer stats');
      return res.json();
    })
    .then(statsData => {
      const metrics = [
        {key:'global_openrank', color:'#4CAF50', label:'Global OpenRank'},
        {key:'community_openrank', color:'#2196F3', label:'Community OpenRank'},
        {key:'avg_activity', color:'#FF9800', label:'Avg Activity'},
      ];

      const chartContainer = document.createElement('div');
      chartContainer.style.marginTop = '12px';
      chartContainer.style.width = '100%';

      const title = document.createElement('h3');
      title.innerText = `Assigned Developer Metrics Display`;
      title.style.marginBottom = '16px';
      title.style.textAlign = 'center';
      chartContainer.appendChild(title);

      // Add legend
      const legendContainer = document.createElement('div');
      legendContainer.style.display = 'flex';
      legendContainer.style.flexWrap = 'wrap';
      legendContainer.style.marginBottom = '10px';
      legendContainer.style.paddingLeft = '40px';

      metrics.forEach(m => {
        const legendItem = document.createElement('div');
        legendItem.style.display = 'flex';
        legendItem.style.alignItems = 'center';
        legendItem.style.marginRight = '15px';

        const colorBox = document.createElement('div');
        colorBox.style.width = '14px';
        colorBox.style.height = '14px';
        colorBox.style.backgroundColor = m.color;
        colorBox.style.marginRight = '5px';

        const labelSpan = document.createElement('span');
        labelSpan.innerText = m.label;
        labelSpan.style.fontSize = '13px';

        legendItem.appendChild(colorBox);
        legendItem.appendChild(labelSpan);
        legendContainer.appendChild(legendItem);
      });
      chartContainer.appendChild(legendContainer);

      // Find the maximum sum to scale bar lengths
      let maxTotal = 0;
      const devDataMap = {};
      recommendation.assignee.forEach(assignee => {
        const devData = statsData.find(d => d.developer === assignee) || {};
        const total = metrics.reduce((sum,m) => sum + (devData[m.key] || 0), 0);
        if (total > maxTotal) maxTotal = total;
        devDataMap[assignee] = {devData, total};
      });

      const baseWidth = 300;

      recommendation.assignee.forEach(assignee => {
        const {devData, total} = devDataMap[assignee];
        const barContainer = document.createElement('div');
        barContainer.style.display = 'flex';
        barContainer.style.alignItems = 'center';
        barContainer.style.marginBottom = '8px';
        barContainer.style.paddingTop = '10px';

        const nameSpan = document.createElement('span');
        nameSpan.innerText = assignee + ' :';
        nameSpan.style.width = '120px';
        nameSpan.style.fontWeight = '600';
        nameSpan.style.color = '#24292e';
        nameSpan.style.fontSize = '14px';
        barContainer.appendChild(nameSpan);

        const barWrapper = document.createElement('div');
        barWrapper.style.display = 'flex';
        barWrapper.style.width = '368px';
        barWrapper.style.height = '25px';
        barWrapper.style.marginLeft = '6px';
        barWrapper.style.background = '#e1e4e8';
        barWrapper.style.position = 'relative';
        barWrapper.style.borderRadius = '4px';
        barWrapper.style.overflow = 'hidden';

        metrics.forEach(m => {
          const val = devData[m.key] || 0;
          if (val <= 0 || total === 0) {
            return;
          }
          const fraction = val / total;
          const segWidth = fraction * baseWidth;

          const seg = document.createElement('div');
          seg.style.width = `${(segWidth / baseWidth) * 100}%`; 
          seg.style.background = m.color;
          seg.style.display = 'flex';
          seg.style.alignItems = 'center';
          seg.style.justifyContent = 'center';
          seg.style.color = '#fff';
          seg.style.fontSize = '12px';
          seg.style.fontWeight = 'bold';
          seg.style.boxSizing = 'border-box';

          // If segment width is too small, hide text, only show color, display on hover
          if ((segWidth / baseWidth) * 100 < 10) { // Hide text when less than 10% width
            seg.innerText = '';
          } else {
            seg.innerText = val.toFixed(2); 
          }

          // Hover tooltip
          seg.title = `${m.label}: ${val.toFixed(2)}`;

          barWrapper.appendChild(seg);
        });

        barContainer.appendChild(barWrapper);
        chartContainer.appendChild(barContainer);
      });

      chartDisplayArea.appendChild(chartContainer);
    })
    .catch(err => {
      console.error('Error fetching developer stats:', err);
    });
  }

  // Run addResolverButton on initial page load and when GitHub uses PJAX navigation (pjax:end event)
  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    addResolverButton();
  } else {
    document.addEventListener('DOMContentLoaded', addResolverButton);
  }

  // When GitHub uses PJAX to load new pages (including switching to issue pages), the pjax:end event is triggered
  document.addEventListener('pjax:end', () => {
    addResolverButton();
  });
})();

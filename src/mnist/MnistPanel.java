package mnist;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.io.IOException;

public class MnistPanel extends JPanel {

    private final MnistDataPoint[] dataPoints;
    private final boolean showLabel = false;
    private int currentIndex = 0;

    public MnistPanel(MnistDataPoint[] dataPoints) {
        super();
        this.dataPoints = dataPoints;
    }

    public static void showImages(MnistDataPoint[] dataPoints) throws IOException, InterruptedException {
        JFrame f = new JFrame();
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.setBounds(100, 100, 450, 300);
        JPanel contentPane = new JPanel();
        contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
        contentPane.setLayout(new BorderLayout(0, 0));

        MnistPanel p = new MnistPanel(dataPoints);
        p.nextImage();

        contentPane.add(p, BorderLayout.CENTER);

        f.setContentPane(contentPane);
        f.setVisible(true);

        for(int i = 0; i < 4000; i++){
            Thread.sleep(500);
            p.nextImage();
            f.repaint();
        }
    }

    private void nextImage() {
        if (currentIndex == dataPoints.length-1)
            currentIndex = 0;
        else
            currentIndex++;
    }

    public void paintComponent(Graphics g) {
        Graphics2D g2d = (Graphics2D) g;

        int w = this.getWidth();
        int h = this.getHeight();

        g.clearRect(0, 0, w, h);

        for(int i = 0; i < 784; i++){
            int x = (i % 28) * w / 28;
            int y = (i / 28) * h / 28;
            Color c = new Color(
                    (int)(dataPoints[currentIndex].inputs[i]*255),
                (int)(dataPoints[currentIndex].inputs[i]*255),
                (int)(dataPoints[currentIndex].inputs[i]*255));
            // g2d.setColor(new Color((int)(dataPoints[currentIndex].inputs[i]*255)));
            g2d.setColor(c);
            g2d.fillRect(x, y, w / 28 + 1, h / 28 + 1);
        }

        if(!showLabel)
            return;

        Composite comp = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, .3f);
        g2d.setComposite(comp);
        g2d.setPaint(Color.red);
        g2d.setFont(new Font("Times Roman", Font.PLAIN, h));
        g2d.drawString(""+dataPoints[currentIndex].label,h / 2, w /2);
    }
}
